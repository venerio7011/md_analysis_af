"""
af3_air_validator_v2.py

AF3 -> HADDOCK interface/AIR validator, redesigned around AF3 full_data JSON.

Design principles
-----------------
1. AF3 full_data JSON is the PRIMARY source for predicted residue-residue contact evidence.
2. The matched AF3 model_0..4.cif/PDB is used for structural/chain/residue consistency checks.
3. summary_confidences_N.json is used for prediction-level confidence/provenance, but it is
   not used as a substitute for residue-level contact evidence.
4. Contact evidence comes from AF3 `contact_probs` together with `token_chain_ids`
   and `token_res_ids`.
5. PDB geometry is VALIDATION evidence, not the source from which interface residues
   are invented.
6. The contact-probability threshold is determined automatically for each prediction
   from the cross-chain AF3 contact-probability distribution.
7. AIR output is explicitly marked as CANDIDATE AIR unless all validation gates pass.
8. No HADDOCK execution is performed by this module.

Expected AF3 full_data schema observed in the supplied file
------------------------------------------------------------
{
    "atom_chain_ids": [...],
    "atom_plddts": [...],
    "contact_probs": [[...], ...],
    "pae": [[...], ...],
    "token_chain_ids": [...],
    "token_res_ids": [...]
}

The supplied example contains 1825 tokens:
    chain A: 1281 tokens/residues
    chain B:  544 tokens/residues

The implementation intentionally does not assume those exact lengths; it validates them
from the JSON at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import argparse
import json
import math
import statistics


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    index: int
    chain_id: str
    residue_id: int


@dataclass(frozen=True)
class ContactPair:
    token_i: int
    token_j: int
    chain_i: str
    residue_i: int
    chain_j: str
    residue_j: int
    contact_probability: float
    pae: Optional[float]


@dataclass(frozen=True)
class InterfaceResidue:
    chain_id: str
    residue_id: int
    max_contact_probability: float
    mean_top_contact_probability: float
    partner_residue_count: int


@dataclass
class ValidationReport:
    status: str
    prediction_id: str
    full_data_json: str
    structure_file: Optional[str]
    summary_file: Optional[str]
    gates: Dict[str, Any]
    json_schema: Dict[str, Any]
    chain_summary: Dict[str, Any]
    adaptive_threshold: Dict[str, Any]
    contact_summary: Dict[str, Any]
    interface_residues: Dict[str, Any]
    structure_consistency: Dict[str, Any]
    air: Dict[str, Any]
    computational_gate: Dict[str, Any]
    downstream: Dict[str, Any]
    evidence_metadata: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


# ---------------------------------------------------------------------------
# Basic I/O
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# AF3 full_data parsing
# ---------------------------------------------------------------------------

REQUIRED_KEYS = (
    "contact_probs",
    "pae",
    "token_chain_ids",
    "token_res_ids",
)


def validate_full_data_schema(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("AF3 full_data must be a JSON object.")

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"Missing required AF3 full_data keys: {missing}")

    chain_ids = data["token_chain_ids"]
    residue_ids = data["token_res_ids"]
    contacts = data["contact_probs"]
    pae = data["pae"]

    if not isinstance(chain_ids, list) or not isinstance(residue_ids, list):
        raise ValueError("token_chain_ids and token_res_ids must be lists.")

    n = len(chain_ids)
    if len(residue_ids) != n:
        raise ValueError(
            f"token_res_ids length {len(residue_ids)} != token_chain_ids length {n}"
        )

    def square_shape(matrix: Any, name: str) -> Tuple[int, int]:
        if not isinstance(matrix, list):
            raise ValueError(f"{name} must be a list.")
        rows = len(matrix)
        if rows != n:
            raise ValueError(
                f"{name} rows {rows} != token count {n}"
            )
        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != n:
                raise ValueError(
                    f"{name}[{i}] has length "
                    f"{len(row) if isinstance(row, list) else 'NON-LIST'}, expected {n}"
                )
        return rows, n

    contact_shape = square_shape(contacts, "contact_probs")
    pae_shape = square_shape(pae, "pae")

    return {
        "required_keys_present": True,
        "token_count": n,
        "contact_probs_shape": list(contact_shape),
        "pae_shape": list(pae_shape),
        "atom_chain_ids_present": "atom_chain_ids" in data,
        "atom_plddts_present": "atom_plddts" in data,
        "schema_status": "PASS",
    }


def build_tokens(data: Dict[str, Any]) -> List[Token]:
    tokens = []
    for i, (chain, residue) in enumerate(
        zip(data["token_chain_ids"], data["token_res_ids"])
    ):
        try:
            residue_i = int(residue)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid token_res_ids[{i}]={residue!r}"
            ) from exc
        tokens.append(Token(i, str(chain), residue_i))
    return tokens


def summarize_chains(tokens: Sequence[Token]) -> Dict[str, Any]:
    by_chain: Dict[str, List[int]] = {}
    for t in tokens:
        by_chain.setdefault(t.chain_id, []).append(t.residue_id)

    result = {}
    for chain, residues in by_chain.items():
        unique = list(dict.fromkeys(residues))
        result[chain] = {
            "token_count": len(residues),
            "unique_residue_count": len(unique),
            "first_residue": min(unique) if unique else None,
            "last_residue": max(unique) if unique else None,
            "residue_ids_are_unique": len(unique) == len(residues),
        }
    return result


# ---------------------------------------------------------------------------
# Cross-chain contact extraction
# ---------------------------------------------------------------------------

def choose_two_chains(tokens: Sequence[Token]) -> Tuple[str, str]:
    chains = sorted({t.chain_id for t in tokens})
    if len(chains) != 2:
        raise ValueError(
            f"Expected exactly two protein chains for this validator; found {chains}"
        )
    return chains[0], chains[1]


def extract_cross_chain_pairs(
    data: Dict[str, Any],
    tokens: Sequence[Token],
    chain_a: str,
    chain_b: str,
    min_probability: float = 0.0,
) -> List[ContactPair]:
    contacts = data["contact_probs"]
    pae = data["pae"]
    pairs: List[ContactPair] = []

    for i, ti in enumerate(tokens):
        if ti.chain_id != chain_a:
            continue
        for j, tj in enumerate(tokens):
            if tj.chain_id != chain_b:
                continue

            p = float(contacts[i][j])
            if not math.isfinite(p):
                continue
            if p < min_probability:
                continue

            q = pae[i][j]
            pae_value = None
            try:
                qf = float(q)
                if math.isfinite(qf):
                    pae_value = qf
            except (TypeError, ValueError):
                pass

            pairs.append(
                ContactPair(
                    token_i=i,
                    token_j=j,
                    chain_i=ti.chain_id,
                    residue_i=ti.residue_id,
                    chain_j=tj.chain_id,
                    residue_j=tj.residue_id,
                    contact_probability=p,
                    pae=pae_value,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------

def _otsu_threshold(values: Sequence[float], bins: int = 256) -> Optional[float]:
    """
    Otsu threshold on [0, 1] probabilities.

    This is used only as an unsupervised diagnostic threshold. It is not a
    trained biological classifier.
    """
    finite = [x for x in values if math.isfinite(x) and 0.0 <= x <= 1.0]
    if len(finite) < 20:
        return None

    lo, hi = min(finite), max(finite)
    if hi <= lo:
        return None

    hist = [0] * bins
    for x in finite:
        idx = min(bins - 1, max(0, int((x - lo) / (hi - lo) * bins)))
        hist[idx] += 1

    total = len(finite)
    sum_total = sum(i * h for i, h in enumerate(hist))
    weight_bg = 0
    sum_bg = 0
    best_var = -1.0
    best_idx = 0

    for i, h in enumerate(hist):
        weight_bg += h
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += i * h
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if between > best_var:
            best_var = between
            best_idx = i

    return lo + (best_idx + 0.5) / bins * (hi - lo)


def _largest_gap_threshold(values: Sequence[float]) -> Optional[float]:
    finite = sorted(
        {round(x, 8) for x in values if math.isfinite(x) and x > 0.0}
    )
    if len(finite) < 5:
        return None

    gaps = [(finite[i + 1] - finite[i], i) for i in range(len(finite) - 1)]
    gap, idx = max(gaps, key=lambda x: x[0])

    # Ignore tiny numerical gaps.
    if gap < 0.03:
        return None

    return (finite[idx] + finite[idx + 1]) / 2.0


def determine_adaptive_threshold(
    cross_chain_pairs: Sequence[ContactPair],
) -> Dict[str, Any]:
    values = [p.contact_probability for p in cross_chain_pairs]
    positive = [x for x in values if x > 0.0]

    if not positive:
        return {
            "status": "NO_SIGNAL",
            "method": "none",
            "threshold": None,
            "reason": "No positive cross-chain contact probabilities.",
            "n_pairs": len(values),
            "n_positive": 0,
        }

    otsu = _otsu_threshold(values)
    gap = _largest_gap_threshold(values)

    # Primary policy:
    # - Otsu is preferred when it yields a biologically usable probability.
    # - The largest distributional gap is used if Otsu is unavailable/uninformative.
    # - The final threshold is bounded to avoid a pathological threshold near 0.
    candidates = []
    if otsu is not None:
        candidates.append(("otsu", otsu))
    if gap is not None:
        candidates.append(("largest_gap", gap))

    usable = [
        (name, value)
        for name, value in candidates
        if 0.20 <= value <= 0.95
    ]

    if usable:
        # Prefer Otsu because it uses the whole distribution.
        chosen_name, chosen = next(
            ((n, v) for n, v in usable if n == "otsu"),
            usable[0],
        )
    else:
        # Weak/ambiguous AF3 interfaces can be quantized at values such as
        # 0.01 and 0.02. Do not impose a fixed 0.20 floor in that case.
        unique_positive = sorted(set(positive))
        if len(unique_positive) <= 10 and max(positive) <= 0.20:
            chosen_name, chosen = "max_observed_discrete_probability", max(positive)
        else:
            q95 = statistics.quantiles(positive, n=20, method="inclusive")[18] \
                if len(positive) >= 20 else max(positive)
            chosen_name, chosen = "upper_tail_q95", q95

    chosen = max(0.0, min(0.95, float(chosen)))

    selected = [p for p in cross_chain_pairs if p.contact_probability >= chosen]

    return {
        "status": "PASS" if selected else "NO_CONTACTS_AT_THRESHOLD",
        "method": chosen_name,
        "threshold": chosen,
        "reason": "Prediction-specific threshold derived from cross-chain AF3 contact-probability distribution.",
        "n_pairs": len(values),
        "n_positive": len(positive),
        "min_positive": min(positive),
        "median_positive": statistics.median(positive),
        "max_positive": max(positive),
        "otsu_threshold": otsu,
        "largest_gap_threshold": gap,
        "selected_pair_count": len(selected),
    }


# ---------------------------------------------------------------------------
# Interface residue extraction
# ---------------------------------------------------------------------------

def extract_interface_residues(
    contacts: Sequence[ContactPair],
    threshold: float,
    top_k_partners: int = 5,
) -> Dict[str, List[InterfaceResidue]]:
    by_residue: Dict[Tuple[str, int], List[ContactPair]] = {}

    for p in contacts:
        if p.contact_probability >= threshold:
            by_residue.setdefault((p.chain_i, p.residue_i), []).append(p)
            by_residue.setdefault((p.chain_j, p.residue_j), []).append(
                ContactPair(
                    token_i=p.token_j,
                    token_j=p.token_i,
                    chain_i=p.chain_j,
                    residue_i=p.residue_j,
                    chain_j=p.chain_i,
                    residue_j=p.residue_i,
                    contact_probability=p.contact_probability,
                    pae=p.pae,
                )
            )

    result: Dict[str, List[InterfaceResidue]] = {}
    for (chain, residue), pairs in by_residue.items():
        pairs = sorted(
            pairs,
            key=lambda x: x.contact_probability,
            reverse=True,
        )
        top = pairs[:top_k_partners]
        result.setdefault(chain, []).append(
            InterfaceResidue(
                chain_id=chain,
                residue_id=residue,
                max_contact_probability=top[0].contact_probability,
                mean_top_contact_probability=statistics.mean(
                    p.contact_probability for p in top
                ),
                partner_residue_count=len(pairs),
            )
        )

    for chain in result:
        result[chain].sort(
            key=lambda x: (
                -x.max_contact_probability,
                x.residue_id,
            )
        )

    return result


# ---------------------------------------------------------------------------
# PDB/mmCIF-lite structural validation
# ---------------------------------------------------------------------------

def parse_pdb_ca_atoms(
    structure_file: str | Path,
) -> Dict[str, Dict[int, Tuple[float, float, float]]]:
    """
    Parse ATOM/HETATM CA coordinates from a PDB file.

    The structure preprocessor in the existing pipeline converts AF3 mmCIF
    to PDB before this validator. This function intentionally consumes the
    resulting PDB so that chain/residue numbering can be checked directly.
    """
    p = Path(structure_file)
    if not p.is_file():
        raise FileNotFoundError(f"Structure file not found: {p}")

    chains: Dict[str, Dict[int, Tuple[float, float, float]]] = {}

    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue

            chain = line[21].strip() or "_"
            try:
                residue = int(line[22:26].strip())
                xyz = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue

            chains.setdefault(chain, {})[residue] = xyz

    return chains


def validate_structure_mapping(
    interface_residues: Dict[str, List[InterfaceResidue]],
    pdb_ca: Dict[str, Dict[int, Tuple[float, float, float]]],
) -> Dict[str, Any]:
    per_chain = {}
    missing_total = 0

    for chain, residues in interface_residues.items():
        pdb_res = pdb_ca.get(chain, {})
        missing = [
            r.residue_id
            for r in residues
            if r.residue_id not in pdb_res
        ]
        missing_total += len(missing)

        per_chain[chain] = {
            "af3_interface_residue_count": len(residues),
            "pdb_ca_residue_count": len(pdb_res),
            "missing_interface_residues_in_pdb": missing,
            "mappable_interface_fraction": (
                (len(residues) - len(missing)) / len(residues)
                if residues else 0.0
            ),
        }

    return {
        "status": "PASS" if missing_total == 0 else "REVIEW",
        "pdb_chain_ids": sorted(pdb_ca),
        "per_chain": per_chain,
        "missing_interface_residue_count": missing_total,
    }


def calculate_structure_distances(
    contacts: Sequence[ContactPair],
    threshold: float,
    pdb_ca: Dict[str, Dict[int, Tuple[float, float, float]]],
) -> Dict[str, Any]:
    distances = []

    for p in contacts:
        if p.contact_probability < threshold:
            continue
        if (
            p.residue_i not in pdb_ca.get(p.chain_i, {})
            or p.residue_j not in pdb_ca.get(p.chain_j, {})
        ):
            continue

        a = pdb_ca[p.chain_i][p.residue_i]
        b = pdb_ca[p.chain_j][p.residue_j]
        d = math.dist(a, b)

        distances.append(
            {
                "chain_i": p.chain_i,
                "residue_i": p.residue_i,
                "chain_j": p.chain_j,
                "residue_j": p.residue_j,
                "contact_probability": p.contact_probability,
                "pae": p.pae,
                "ca_distance_angstrom": d,
            }
        )

    if not distances:
        return {
            "status": "NO_MAPPABLE_CONTACTS",
            "count": 0,
            "distance_summary": {},
            "pairs": [],
        }

    vals = [x["ca_distance_angstrom"] for x in distances]
    return {
        "status": "PASS",
        "count": len(vals),
        "distance_summary": {
            "min": min(vals),
            "median": statistics.median(vals),
            "max": max(vals),
            "within_8A": sum(d <= 8.0 for d in vals),
            "within_10A": sum(d <= 10.0 for d in vals),
            "within_12A": sum(d <= 12.0 for d in vals),
        },
        "pairs": sorted(
            distances,
            key=lambda x: (
                -x["contact_probability"],
                x["ca_distance_angstrom"],
            ),
        ),
    }


# ---------------------------------------------------------------------------
# AIR generation
# ---------------------------------------------------------------------------

def _air_line(chain_a: str, res_a: int, chain_b: str, res_b: int) -> str:
    # HADDOCK restraint syntax: chain identifier is represented as segid.
    # Bounds are deliberately modest and are NOT claimed to be experimentally
    # calibrated. They are candidate restraints for downstream review.
    return (
        f"assign (resid {res_a} and segid {chain_a}) "
        f"(resid {res_b} and segid {chain_b}) 2.0 2.0 0.0"
    )


def generate_candidate_air(
    selected_pairs: Sequence[ContactPair],
    output_path: str | Path,
    max_pairs: int = 50,
) -> Dict[str, Any]:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    unique = {}
    for c in selected_pairs:
        key = (
            c.chain_i,
            c.residue_i,
            c.chain_j,
            c.residue_j,
        )
        unique[key] = c

    ranked = sorted(
        unique.values(),
        key=lambda x: (
            -x.contact_probability,
            x.pae if x.pae is not None else float("inf"),
        ),
    )[:max_pairs]

    lines = [
        "! AF3 -> HADDOCK candidate AIR",
        "! Generated from AF3 full_data.json contact_probs.",
        "! Candidate restraints require downstream scientific review.",
        "! This file is NOT experimental ground truth.",
    ]

    for c in ranked:
        lines.append(
            _air_line(
                c.chain_i,
                c.residue_i,
                c.chain_j,
                c.residue_j,
            )
        )

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "file": str(p),
        "status": "PASS" if ranked else "REVIEW",
        "candidate_pair_count": len(ranked),
        "production_ready": False,
        "pairs": [
            {
                "chain_i": c.chain_i,
                "residue_i": c.residue_i,
                "chain_j": c.chain_j,
                "residue_j": c.residue_j,
                "contact_probability": c.contact_probability,
                "pae": c.pae,
            }
            for c in ranked
        ],
    }


# ---------------------------------------------------------------------------
# Optional summary_confidences reader
# ---------------------------------------------------------------------------

def inspect_summary_confidence(
    summary_file: Optional[str | Path],
) -> Dict[str, Any]:
    if summary_file is None:
        return {"status": "NOT_PROVIDED"}

    data = load_json(summary_file)
    fields = (
        "ranking_score",
        "iptm",
        "ptm",
        "chain_iptm",
        "chain_ptm",
        "chain_pair_iptm",
        "chain_pair_pae_min",
        "chain_ids",
    )

    result = {"status": "PASS", "file": str(summary_file)}
    for key in fields:
        if key in data:
            value = data[key]
            if key == "chain_ids" and isinstance(value, list):
                value = list(dict.fromkeys(str(x) for x in value))
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def validate_prediction(
    full_data_json: str | Path,
    structure_file: Optional[str | Path],
    output_directory: str | Path,
    summary_file: Optional[str | Path] = None,
    max_air_pairs: int = 50,
    pae_gate: Optional[float] = 30.0,
) -> ValidationReport:
    full_path = Path(full_data_json)
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)

    prediction_id = full_path.stem.replace("_full_data_0", "")
    data = load_json(full_path)

    warnings: List[str] = []
    errors: List[str] = []

    # Gate 1: schema
    try:
        schema = validate_full_data_schema(data)
        tokens = build_tokens(data)
    except Exception as exc:
        errors.append(f"JSON schema/token validation failed: {exc}")
        report = ValidationReport(
            status="FAILED",
            prediction_id=prediction_id,
            full_data_json=str(full_path),
            structure_file=str(structure_file) if structure_file else None,
            summary_file=str(summary_file) if summary_file else None,
            gates={"json_schema": "FAIL"},
            json_schema={},
            chain_summary={},
            adaptive_threshold={},
            contact_summary={},
            interface_residues={},
            structure_consistency={},
            air={},
            computational_gate={"status": "FAIL", "hard_gate": True, "reasons": errors, "air_required": False, "pae_validated_air_required": False},
            downstream={"action": "STOP", "haddock_mode": None, "air_available": False, "air_required": False},
            evidence_metadata={},
            warnings=warnings,
            errors=errors,
        )
        write_json(out / f"{prediction_id}_validator_report.json", asdict(report))
        return report

    # Gate 2: chain topology
    try:
        chain_a, chain_b = choose_two_chains(tokens)
    except Exception as exc:
        errors.append(str(exc))
        chain_a = chain_b = ""

    chain_summary = summarize_chains(tokens)

    # Gate 3: AF3 contact evidence
    cross_pairs: List[ContactPair] = []
    adaptive: Dict[str, Any] = {}

    if not errors:
        cross_pairs = extract_cross_chain_pairs(
            data, tokens, chain_a, chain_b, min_probability=0.0
        )
        adaptive = determine_adaptive_threshold(cross_pairs)

    threshold = adaptive.get("threshold")
    contact_selected_pairs: List[ContactPair] = []
    pae_pass_pairs: List[ContactPair] = []

    if threshold is not None:
        contact_selected_pairs = [p for p in cross_pairs if p.contact_probability >= threshold]
        pae_pass_pairs = [
            p for p in contact_selected_pairs
            if pae_gate is None or p.pae is None or p.pae <= pae_gate
        ]

    # AF3 contact_probs defines residue evidence. PAE is a validation gate
    # for candidate restraints, not the source of residue identities.
    interface = (
        extract_interface_residues(contact_selected_pairs, threshold)
        if threshold is not None else {}
    )

    # Contact summary
    contact_summary = {
        "chain_pair": [chain_a, chain_b] if chain_a and chain_b else [],
        "cross_chain_pair_count": len(cross_pairs),
        "selected_pair_count_after_threshold_and_pae_gate": len(contact_selected_pairs),
        "pae_gate_angstrom": pae_gate,
        "top_selected_pairs": [
            {
                "chain_i": p.chain_i,
                "residue_i": p.residue_i,
                "chain_j": p.chain_j,
                "residue_j": p.residue_j,
                "contact_probability": p.contact_probability,
                "pae": p.pae,
            }
            for p in sorted(
                pae_pass_pairs,
                key=lambda x: (
                    -x.contact_probability,
                    x.pae if x.pae is not None else float("inf"),
                ),
            )[:100]
        ],
    }

    # Gate 4: structure mapping/geometry
    structure_consistency: Dict[str, Any] = {
        "status": "NOT_PROVIDED"
    }

    if structure_file:
        try:
            pdb_ca = parse_pdb_ca_atoms(structure_file)
            structure_consistency["mapping"] = validate_structure_mapping(
                interface, pdb_ca
            )
            structure_consistency["geometry"] = calculate_structure_distances(
                cross_pairs,
                threshold if threshold is not None else 1.0,
                pdb_ca,
            )
            structure_consistency["status"] = (
                "PASS"
                if structure_consistency["mapping"]["status"] == "PASS"
                else "REVIEW"
            )
        except Exception as exc:
            structure_consistency = {
                "status": "FAILED",
                "error": str(exc),
            }
            errors.append(f"Structure validation failed: {exc}")

    # Gate 5: AIR
    air_path = out / f"{prediction_id}_candidate_air.tbl"
    air = generate_candidate_air(
        pae_pass_pairs,
        air_path,
        max_pairs=max_air_pairs,
    )

    if not contact_selected_pairs:
        warnings.append("No AF3 residue pairs passed the adaptive contact-probability threshold.")
    elif not pae_pass_pairs:
        warnings.append("AF3 contact evidence exists, but no selected pairs passed the PAE validation gate.")

    if structure_file and structure_consistency.get("status") == "FAILED":
        warnings.append(
            "AIR was generated from AF3 JSON evidence, but structural validation failed."
        )

    if not interface:
        warnings.append(
            "No interface residues were identified from AF3 contact_probs."
        )

    # Computational integrity is the hard gate. AIR/PAE is optional evidence.
    computational_reasons = []
    if not chain_a or not chain_b:
        computational_reasons.append("AF3 model does not contain exactly two chains.")
    if not structure_file:
        computational_reasons.append("Selected PDB is required for downstream HADDOCK execution.")
    elif structure_consistency.get("status") != "PASS":
        computational_reasons.append("Selected PDB failed structural consistency validation.")
    if errors:
        computational_reasons.extend(errors)

    computational_ok = len(computational_reasons) == 0
    computational_gate = {
        "status": "PASS" if computational_ok else "FAIL",
        "hard_gate": True,
        "reasons": computational_reasons,
        "air_required": False,
        "pae_validated_air_required": False,
    }

    # AIR is optional. PAE failure/no AIR does not stop HADDOCK.
    if pae_pass_pairs and computational_ok:
        air["status"] = "ACTIVE"
        air["active"] = True
        air["production_ready"] = True
        air["path"] = air["file"]
        downstream = {
            "action": "RUN_HADDOCK",
            "haddock_mode": "AF3_DERIVED",
            "air_available": True,
            "air_required": False,
        }
    else:
        air["status"] = "NONE"
        air["active"] = False
        air["production_ready"] = False
        air["path"] = None
        rejected_path = out / f"{prediction_id}_candidate_air.REJECTED.tbl"
        rejected_path.write_text(
            "! AF3 -> HADDOCK candidate AIR\n"
            "! STATUS: REJECTED / NONE\n"
            "! No PAE-qualified candidate restraints are available.\n",
            encoding="utf-8",
        )
        air["rejected_path"] = str(rejected_path)
        air["rejected_reason"] = (
            "NO_PAE_VALIDATED_AIR" if contact_selected_pairs else "NO_CONTACT_CANDIDATES"
        )
        downstream = (
            {
                "action": "RUN_HADDOCK",
                "haddock_mode": "NONE",
                "air_available": False,
                "air_required": False,
            }
            if computational_ok else
            {
                "action": "STOP",
                "haddock_mode": None,
                "air_available": False,
                "air_required": False,
            }
        )

    status = "PASS" if computational_ok else "FAILED"

    gates = {
        "json_schema": "PASS",
        "two_chain_model": "PASS" if chain_a and chain_b else "FAIL",
        "af3_contact_evidence": "PASS" if contact_selected_pairs else "NONE",
        "pae_gate": "PASS" if pae_pass_pairs else "NONE",
        "structure_consistency": structure_consistency.get("status", "NOT_PROVIDED"),
        "candidate_air_generation": air["status"],
        "candidate_air_active": "PASS" if air.get("active") else "NONE",
        "computational_gate": computational_gate["status"],
        "downstream_action": downstream["action"],
        "final": status,
    }

    interface_serialized = {
        chain: [asdict(x) for x in residues]
        for chain, residues in interface.items()
    }

    evidence_metadata = {
        "metadata_version": "2.1",
        "prediction_id": prediction_id,
        "sources": {
            "full_data_json": str(full_path),
            "structure_file": str(structure_file) if structure_file else None,
            "summary_file": str(summary_file) if summary_file else None,
        },
        "schema": schema,
        "chain_summary": chain_summary,
        "adaptive_threshold": adaptive,
        "contact_evidence": {
            "cross_chain_pair_count": len(cross_pairs),
            "candidate_pair_count": len(contact_selected_pairs),
            "pae_validated_pair_count": len(pae_pass_pairs),
        },
        "interface_residues": interface_serialized,
        "structure_consistency": structure_consistency,
        "air": air,
        "computational_gate": computational_gate,
        "downstream": downstream,
        "warnings": warnings,
        "errors": errors,
    }

    report = ValidationReport(
        status=status,
        prediction_id=prediction_id,
        full_data_json=str(full_path),
        structure_file=str(structure_file) if structure_file else None,
        summary_file=str(summary_file) if summary_file else None,
        gates=gates,
        json_schema=schema,
        chain_summary=chain_summary,
        adaptive_threshold=adaptive,
        contact_summary=contact_summary,
        interface_residues=interface_serialized,
        structure_consistency=structure_consistency,
        air=air,
        computational_gate=computational_gate,
        downstream=downstream,
        evidence_metadata=evidence_metadata,
        warnings=warnings,
        errors=errors,
    )

    write_json(
        out / f"{prediction_id}_validator_report.json",
        asdict(report),
    )

    # Save a compact interface-residue file for downstream inspection.
    write_json(
        out / f"{prediction_id}_interface_residues.json",
        {
            "prediction_id": prediction_id,
            "threshold": threshold,
            "interface_residues": interface_serialized,
        },
    )

    write_json(out / f"{prediction_id}_evidence_metadata.json", evidence_metadata)
    write_json(out / f"{prediction_id}_downstream_policy.json", downstream)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate AF3 full_data JSON and generate candidate HADDOCK AIR."
    )
    parser.add_argument("--full-data", required=True)
    parser.add_argument("--structure", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-air-pairs", type=int, default=50)
    parser.add_argument(
        "--pae-gate",
        type=float,
        default=30.0,
        help="Maximum AF3 PAE allowed for a candidate pair; use -1 to disable.",
    )

    args = parser.parse_args()
    pae_gate = None if args.pae_gate < 0 else args.pae_gate

    report = validate_prediction(
        full_data_json=args.full_data,
        structure_file=args.structure,
        summary_file=args.summary,
        output_directory=args.output,
        max_air_pairs=args.max_air_pairs,
        pae_gate=pae_gate,
    )

    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    return 0 if report.status != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
