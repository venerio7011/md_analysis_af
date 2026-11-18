#!/usr/bin/env python3

"""
structure_preprocessor.py

AF3 → HADDOCK structure preprocessing module.

Responsibilities
----------------
1. Discover AF3 model_*.cif files.
2. Ignore template CIF files.
3. Match each model_N.cif with summary_confidences_N.json.
4. Select the best AF3 model.
5. Convert the selected mmCIF structure to PDB.
6. Validate the generated PDB.
7. Validate expected protein-protein chains.
8. Preserve AF3 confidence/provenance information.
9. Generate per-fold and batch manifests.

This module does NOT run HADDOCK or OpenMM.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import gemmi
except ImportError:
    gemmi = None


# ============================================================
# Data structures
# ============================================================

@dataclass
class ModelCandidate:

    model_index: int

    cif_path: str

    confidence_path: str

    iptm: Optional[float]

    ptm: Optional[float]

    ranking_score: Optional[float]

    fraction_disordered: Optional[float]

    has_clash: Optional[float]

    chain_ids: List[str]

    chain_pair_iptm: Any

    chain_pair_pae_min: Any

    chain_ptm: Any


@dataclass
class StructureInfo:

    fold_id: str

    selected_model_index: Optional[int]

    selected_model_cif: str

    output_pdb: str

    chains: List[str]

    residue_count: int

    atom_count: int

    status: str

    selection_reason: str

    selected_iptm: Optional[float]

    selected_ptm: Optional[float]

    selected_ranking_score: Optional[float]

    selected_fraction_disordered: Optional[float]

    selected_has_clash: Optional[float]

    selected_chain_pair_iptm: Any

    selected_chain_pair_pae_min: Any

    model_candidates: List[Dict[str, Any]]

    message: str = ""


# ============================================================
# Dependency
# ============================================================

def check_dependencies() -> None:

    if gemmi is None:

        raise ImportError(
            "\n"
            "Gemmi is required for mmCIF → PDB conversion.\n\n"
            "Install it in Colab with:\n\n"
            "!pip install gemmi\n"
        )


# ============================================================
# Fold ID
# ============================================================

def extract_fold_number(
    fold_name: str
) -> Optional[int]:

    match = re.match(
        r"^fold_(\d+)$",
        fold_name
    )

    if match is None:
        return None

    return int(match.group(1))


# ============================================================
# AF3 model discovery
# ============================================================

def discover_model_files(
    fold_directory: str | Path
) -> List[Tuple[int, Path]]:

    """
    Discover only AF3 model_N.cif files located directly
    inside the fold directory.

    Template CIF files are intentionally excluded.
    """

    fold_directory = Path(
        fold_directory
    )

    if not fold_directory.exists():

        raise FileNotFoundError(
            f"Fold directory does not exist:\n"
            f"{fold_directory}"
        )

    candidates = []

    for path in fold_directory.glob(
        "*_model_*.cif"
    ):

        if not path.is_file():
            continue

        match = re.search(
            r"_model_(\d+)\.cif$",
            path.name
        )

        if match is None:
            continue

        model_index = int(
            match.group(1)
        )

        candidates.append(
            (model_index, path)
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    return candidates


# ============================================================
# Confidence JSON discovery
# ============================================================

def find_confidence_file(
    fold_directory: Path,
    model_index: int
) -> Optional[Path]:

    """
    Find the summary_confidences_N.json corresponding
    to model_N.cif.
    """

    matches = list(
        fold_directory.glob(
            f"*_summary_confidences_{model_index}.json"
        )
    )

    if len(matches) == 0:
        return None

    if len(matches) > 1:

        raise RuntimeError(
            "Multiple confidence files found for "
            f"model_{model_index}:\n"
            + "\n".join(
                str(x)
                for x in matches
            )
        )

    return matches[0]


# ============================================================
# JSON loading
# ============================================================

def load_json(
    path: Path
) -> Dict[str, Any]:

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# Numeric helper
# ============================================================

def safe_float(
    value: Any
) -> Optional[float]:

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return None


# ============================================================
# Model candidate construction
# ============================================================

def build_model_candidate(
    model_index: int,
    cif_path: Path,
    confidence_path: Path
) -> ModelCandidate:

    data = load_json(
        confidence_path
    )

    return ModelCandidate(
        model_index=model_index,

        cif_path=str(
            cif_path
        ),

        confidence_path=str(
            confidence_path
        ),

        iptm=safe_float(
            data.get("iptm")
        ),

        ptm=safe_float(
            data.get("ptm")
        ),

        ranking_score=safe_float(
            data.get("ranking_score")
        ),

        fraction_disordered=safe_float(
            data.get("fraction_disordered")
        ),

        has_clash=safe_float(
            data.get("has_clash")
        ),

        chain_ids=data.get(
            "chain_ids",
            []
        ),

        chain_pair_iptm=data.get(
            "chain_pair_iptm"
        ),

        chain_pair_pae_min=data.get(
            "chain_pair_pae_min"
        ),

        chain_ptm=data.get(
            "chain_ptm"
        )
    )


# ============================================================
# Candidate discovery
# ============================================================

def discover_model_candidates(
    fold_directory: str | Path
) -> List[ModelCandidate]:

    fold_directory = Path(
        fold_directory
    )

    model_files = discover_model_files(
        fold_directory
    )

    if len(model_files) == 0:

        raise FileNotFoundError(
            "No AF3 model_*.cif files found in:\n"
            f"{fold_directory}"
        )

    candidates = []

    for model_index, cif_path in model_files:

        confidence_path = find_confidence_file(
            fold_directory,
            model_index
        )

        if confidence_path is None:

            raise FileNotFoundError(
                "Missing confidence file for "
                f"model_{model_index}:\n"
                f"Expected *_summary_confidences_"
                f"{model_index}.json"
            )

        candidate = build_model_candidate(
            model_index=model_index,
            cif_path=cif_path,
            confidence_path=confidence_path
        )

        candidates.append(
            candidate
        )

    return candidates


# ============================================================
# Best model selection
# ============================================================

def selection_key(
    candidate: ModelCandidate
):
    """
    Ranking strategy:

    1. Higher ipTM
    2. Higher ranking_score
    3. Higher pTM
    4. Lower fraction_disordered
    5. Lower clash
    6. Lower model index

    Missing values are treated as worse.
    """

    iptm = (
        candidate.iptm
        if candidate.iptm is not None
        else float("-inf")
    )

    ranking_score = (
        candidate.ranking_score
        if candidate.ranking_score is not None
        else float("-inf")
    )

    ptm = (
        candidate.ptm
        if candidate.ptm is not None
        else float("-inf")
    )

    fraction_disordered = (
        candidate.fraction_disordered
        if candidate.fraction_disordered is not None
        else float("inf")
    )

    clash = (
        candidate.has_clash
        if candidate.has_clash is not None
        else float("inf")
    )

    return (
        iptm,
        ranking_score,
        ptm,
        -fraction_disordered,
        -clash,
        -candidate.model_index
    )


def select_best_model(
    candidates: List[ModelCandidate]
) -> ModelCandidate:

    if len(candidates) == 0:

        raise ValueError(
            "No model candidates available."
        )

    return max(
        candidates,
        key=selection_key
    )


# ============================================================
# mmCIF → PDB
# ============================================================

def convert_cif_to_pdb(
    cif_path: str | Path,
    pdb_path: str | Path
) -> Path:

    check_dependencies()

    cif_path = Path(
        cif_path
    )

    pdb_path = Path(
        pdb_path
    )

    if not cif_path.exists():

        raise FileNotFoundError(
            f"mmCIF file not found:\n"
            f"{cif_path}"
        )

    pdb_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    structure = gemmi.read_structure(
        str(cif_path)
    )

    structure.write_pdb(
        str(pdb_path)
    )

    if not pdb_path.exists():

        raise RuntimeError(
            "PDB conversion failed:\n"
            f"{pdb_path}"
        )

    if pdb_path.stat().st_size == 0:

        raise RuntimeError(
            "Generated PDB is empty:\n"
            f"{pdb_path}"
        )

    return pdb_path


# ============================================================
# PDB validation
# ============================================================

def validate_pdb(
    pdb_path: str | Path,
    expected_chain_count: int = 2
) -> Dict[str, Any]:

    check_dependencies()

    pdb_path = Path(
        pdb_path
    )

    if not pdb_path.exists():

        return {
            "valid": False,
            "message": "PDB file does not exist.",
            "chains": [],
            "residue_count": 0,
            "atom_count": 0
        }

    try:

        structure = gemmi.read_structure(
            str(pdb_path)
        )

    except Exception as exc:

        return {
            "valid": False,
            "message": (
                f"PDB parsing failed: {exc}"
            ),
            "chains": [],
            "residue_count": 0,
            "atom_count": 0
        }

    chains = set()

    residue_count = 0
    atom_count = 0

    for model in structure:

        for chain in model:

            chains.add(
                chain.name
            )

            for residue in chain:

                residue_count += 1

                for atom in residue:

                    atom_count += 1

    chains = sorted(
        chains
    )

    if atom_count == 0:

        return {
            "valid": False,
            "message": "PDB contains no atoms.",
            "chains": chains,
            "residue_count": residue_count,
            "atom_count": atom_count
        }

    if expected_chain_count is not None:

        if len(chains) != expected_chain_count:

            return {
                "valid": False,
                "message": (
                    f"Expected "
                    f"{expected_chain_count} chains "
                    f"but found {len(chains)}: "
                    f"{chains}"
                ),
                "chains": chains,
                "residue_count": residue_count,
                "atom_count": atom_count
            }

    return {
        "valid": True,
        "message": (
            "PDB successfully parsed."
        ),
        "chains": chains,
        "residue_count": residue_count,
        "atom_count": atom_count
    }


# ============================================================
# Protein chain validation
# ============================================================

def validate_protein_chains(
    pdb_path: str | Path,
    expected_chains: Optional[List[str]] = None
) -> Dict[str, Any]:

    check_dependencies()

    structure = gemmi.read_structure(
        str(pdb_path)
    )

    observed_chains = []

    for model in structure:

        for chain in model:

            if chain.name not in observed_chains:

                observed_chains.append(
                    chain.name
                )

    observed_chains = sorted(
        observed_chains
    )

    result = {
        "valid": True,
        "observed_chains": observed_chains,
        "message": "Chain validation passed."
    }

    if expected_chains is not None:

        expected = sorted(
            expected_chains
        )

        if observed_chains != expected:

            result["valid"] = False

            result["message"] = (
                "Unexpected chain set. "
                f"Expected={expected}, "
                f"Observed={observed_chains}"
            )

    return result


# ============================================================
# Candidate serialization
# ============================================================

def candidate_to_dict(
    candidate: ModelCandidate
) -> Dict[str, Any]:

    return asdict(
        candidate
    )


# ============================================================
# Resume / SKIP support
# ============================================================

def _load_completed_structure(
    output_directory: str | Path,
    fold_id: str,
) -> Optional[StructureInfo]:
    """Return a previously completed structure result when it is valid.

    SKIP is granted only when:
      1. structure_manifest.json exists and is readable;
      2. manifest status is PASS;
      3. the recorded PDB exists;
      4. the PDB is still structurally valid.

    FAILED/INVALID/incomplete artifacts are never treated as completed work.
    """
    output_directory = Path(output_directory)
    manifest_path = output_directory / "structure_manifest.json"

    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if manifest.get("status") != "PASS":
        return None

    pdb_info = manifest.get("pdb", {})
    pdb_path = Path(pdb_info.get("path", ""))
    if not pdb_path.is_file():
        return None

    try:
        validation = validate_pdb(pdb_path)
    except Exception:
        return None

    if not validation.get("valid", False):
        return None

    selected = manifest.get("selected_model", {})
    return StructureInfo(
        fold_id=str(manifest.get("fold_id", fold_id)),
        selected_model_index=selected.get("model_index"),
        selected_model_cif=str(selected.get("cif_path", "")),
        output_pdb=str(pdb_path),
        chains=list(pdb_info.get("chains", validation.get("chains", []))),
        residue_count=int(pdb_info.get("residue_count", validation.get("residue_count", 0))),
        atom_count=int(pdb_info.get("atom_count", validation.get("atom_count", 0))),
        status="SKIPPED",
        selection_reason=str(manifest.get("selection_reason", "Previously completed and validated.")),
        selected_iptm=selected.get("iptm"),
        selected_ptm=selected.get("ptm"),
        selected_ranking_score=selected.get("ranking_score"),
        selected_fraction_disordered=selected.get("fraction_disordered"),
        selected_has_clash=selected.get("has_clash"),
        selected_chain_pair_iptm=selected.get("chain_pair_iptm"),
        selected_chain_pair_pae_min=selected.get("chain_pair_pae_min"),
        model_candidates=manifest.get("all_model_candidates", []),
        message="SKIP: previously completed PASS artifact was found and revalidated.",
    )


# ============================================================
# Single fold processing
# ============================================================

def process_fold(
    fold_directory: str | Path,
    output_directory: str | Path,
    expected_chains: Optional[List[str]] = None,
    skip_completed: bool = True,
) -> StructureInfo:

    fold_directory = Path(
        fold_directory
    )

    output_directory = Path(
        output_directory
    )

    fold_id = fold_directory.name

    # --------------------------------------------------------
    # Resume / SKIP
    # --------------------------------------------------------
    if skip_completed:
        completed = _load_completed_structure(
            output_directory=output_directory,
            fold_id=fold_id,
        )
        if completed is not None:
            print(f"SKIP: {fold_id} — validated structure artifact already exists.")
            return completed

    # --------------------------------------------------------
    # Discover AF3 models
    # --------------------------------------------------------

    try:

        candidates = discover_model_candidates(
            fold_directory
        )

    except Exception as exc:

        return StructureInfo(
            fold_id=fold_id,
            selected_model_index=None,
            selected_model_cif="",
            output_pdb="",
            chains=[],
            residue_count=0,
            atom_count=0,
            status="FAILED",
            selection_reason="",
            selected_iptm=None,
            selected_ptm=None,
            selected_ranking_score=None,
            selected_fraction_disordered=None,
            selected_has_clash=None,
            selected_chain_pair_iptm=None,
            selected_chain_pair_pae_min=None,
            model_candidates=[],
            message=str(exc)
        )

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    selected = select_best_model(
        candidates
    )

    candidate_records = [
        candidate_to_dict(c)
        for c in candidates
    ]

    # --------------------------------------------------------
    # Selection rationale
    # --------------------------------------------------------

    selection_reason = (
        "Selected using the AF3 model ranking strategy: "
        "higher ipTM, then higher ranking_score, "
        "then higher pTM, then lower fraction_disordered, "
        "then lower clash, then lower model index."
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    pdb_path = (
        output_directory /
        f"{fold_id}_model_{selected.model_index}.pdb"
    )

    # --------------------------------------------------------
    # Convert selected model
    # --------------------------------------------------------

    try:

        convert_cif_to_pdb(
            selected.cif_path,
            pdb_path
        )

    except Exception as exc:

        return StructureInfo(
            fold_id=fold_id,
            selected_model_index=selected.model_index,
            selected_model_cif=selected.cif_path,
            output_pdb=str(pdb_path),
            chains=[],
            residue_count=0,
            atom_count=0,
            status="FAILED",
            selection_reason=selection_reason,
            selected_iptm=selected.iptm,
            selected_ptm=selected.ptm,
            selected_ranking_score=selected.ranking_score,
            selected_fraction_disordered=selected.fraction_disordered,
            selected_has_clash=selected.has_clash,
            selected_chain_pair_iptm=selected.chain_pair_iptm,
            selected_chain_pair_pae_min=selected.chain_pair_pae_min,
            model_candidates=candidate_records,
            message=(
                f"Conversion failed: {exc}"
            )
        )

    # --------------------------------------------------------
    # PDB validation
    # --------------------------------------------------------

    validation = validate_pdb(
        pdb_path,
        expected_chain_count=2
    )

    if not validation["valid"]:

        return StructureInfo(
            fold_id=fold_id,
            selected_model_index=selected.model_index,
            selected_model_cif=selected.cif_path,
            output_pdb=str(pdb_path),
            chains=validation["chains"],
            residue_count=validation[
                "residue_count"
            ],
            atom_count=validation[
                "atom_count"
            ],
            status="INVALID",
            selection_reason=selection_reason,
            selected_iptm=selected.iptm,
            selected_ptm=selected.ptm,
            selected_ranking_score=selected.ranking_score,
            selected_fraction_disordered=selected.fraction_disordered,
            selected_has_clash=selected.has_clash,
            selected_chain_pair_iptm=selected.chain_pair_iptm,
            selected_chain_pair_pae_min=selected.chain_pair_pae_min,
            model_candidates=candidate_records,
            message=validation[
                "message"
            ]
        )

    # --------------------------------------------------------
    # Chain validation
    # --------------------------------------------------------

    chain_validation = validate_protein_chains(
        pdb_path,
        expected_chains=expected_chains
    )

    if not chain_validation["valid"]:

        return StructureInfo(
            fold_id=fold_id,
            selected_model_index=selected.model_index,
            selected_model_cif=selected.cif_path,
            output_pdb=str(pdb_path),
            chains=validation["chains"],
            residue_count=validation[
                "residue_count"
            ],
            atom_count=validation[
                "atom_count"
            ],
            status="INVALID",
            selection_reason=selection_reason,
            selected_iptm=selected.iptm,
            selected_ptm=selected.ptm,
            selected_ranking_score=selected.ranking_score,
            selected_fraction_disordered=selected.fraction_disordered,
            selected_has_clash=selected.has_clash,
            selected_chain_pair_iptm=selected.chain_pair_iptm,
            selected_chain_pair_pae_min=selected.chain_pair_pae_min,
            model_candidates=candidate_records,
            message=chain_validation[
                "message"
            ]
        )

    # --------------------------------------------------------
    # Per-fold manifest
    # --------------------------------------------------------

    fold_manifest = {
        "fold_id": fold_id,
        "status": "PASS",

        "selected_model": {
            "model_index": selected.model_index,
            "cif_path": selected.cif_path,
            "confidence_path": selected.confidence_path,
            "iptm": selected.iptm,
            "ptm": selected.ptm,
            "ranking_score": selected.ranking_score,
            "fraction_disordered": (
                selected.fraction_disordered
            ),
            "has_clash": selected.has_clash,
            "chain_ids": selected.chain_ids,
            "chain_pair_iptm": (
                selected.chain_pair_iptm
            ),
            "chain_pair_pae_min": (
                selected.chain_pair_pae_min
            ),
            "chain_ptm": selected.chain_ptm
        },

        "pdb": {
            "path": str(pdb_path),
            "chains": validation["chains"],
            "residue_count": validation[
                "residue_count"
            ],
            "atom_count": validation[
                "atom_count"
            ]
        },

        "selection_reason": selection_reason,

        "all_model_candidates": candidate_records
    }

    manifest_path = (
        output_directory /
        "structure_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            fold_manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    return StructureInfo(
        fold_id=fold_id,
        selected_model_index=selected.model_index,
        selected_model_cif=selected.cif_path,
        output_pdb=str(pdb_path),
        chains=validation["chains"],
        residue_count=validation[
            "residue_count"
        ],
        atom_count=validation[
            "atom_count"
        ],
        status="PASS",
        selection_reason=selection_reason,
        selected_iptm=selected.iptm,
        selected_ptm=selected.ptm,
        selected_ranking_score=selected.ranking_score,
        selected_fraction_disordered=selected.fraction_disordered,
        selected_has_clash=selected.has_clash,
        selected_chain_pair_iptm=(
            selected.chain_pair_iptm
        ),
        selected_chain_pair_pae_min=(
            selected.chain_pair_pae_min
        ),
        model_candidates=candidate_records,
        message=(
            "AF3 model selected, converted to PDB, "
            "and validated successfully."
        )
    )


# ============================================================
# Batch processing
# ============================================================

def process_extracted_folds(
    extracted_directory: str | Path,
    output_directory: str | Path,
    start: Optional[int] = None,
    end: Optional[int] = None,
    expected_chains: Optional[List[str]] = None,
    skip_completed: bool = True,
) -> List[StructureInfo]:

    extracted_directory = Path(
        extracted_directory
    )

    output_directory = Path(
        output_directory
    )

    if not extracted_directory.exists():

        raise FileNotFoundError(
            "Extracted directory does not exist:\n"
            f"{extracted_directory}"
        )

    fold_directories = []

    for path in extracted_directory.iterdir():

        if not path.is_dir():
            continue

        fold_number = extract_fold_number(
            path.name
        )

        if fold_number is None:
            continue

        if start is not None:

            if fold_number < start:
                continue

        if end is not None:

            if fold_number > end:
                continue

        fold_directories.append(
            (fold_number, path)
        )

    fold_directories.sort(
        key=lambda x: x[0]
    )

    if len(fold_directories) == 0:

        raise FileNotFoundError(
            "No fold_NNN directories found in:\n"
            f"{extracted_directory}"
        )

    results = []

    for _, fold_directory in fold_directories:

        print(
            "\n" + "=" * 70
        )

        print(
            f"Processing {fold_directory.name}"
        )

        result = process_fold(
            fold_directory=fold_directory,
            output_directory=(
                output_directory /
                fold_directory.name
            ),
            expected_chains=expected_chains,
            skip_completed=skip_completed,
        )

        results.append(
            result
        )

        print(
            f"Status: {result.status}"
        )

        if result.selected_model_index is not None:

            print(
                "Selected model: "
                f"model_{result.selected_model_index}"
            )

        if result.selected_iptm is not None:

            print(
                f"ipTM: {result.selected_iptm}"
            )

        if result.output_pdb:

            print(
                f"PDB: {result.output_pdb}"
            )

        if result.chains:

            print(
                f"Chains: {result.chains}"
            )

        if result.message:

            print(
                f"Message: {result.message}"
            )

    # --------------------------------------------------------
    # Batch manifest
    # --------------------------------------------------------

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    batch_manifest = [
        asdict(result)
        for result in results
    ]

    manifest_path = (
        output_directory /
        "structure_processing_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            batch_manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    pass_count = sum(
        r.status == "PASS"
        for r in results
    )

    failed_count = sum(
        r.status == "FAILED"
        for r in results
    )

    invalid_count = sum(
        r.status == "INVALID"
        for r in results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STRUCTURE PROCESSING COMPLETE"
    )

    print(
        f"Total: {len(results)}"
    )

    print(
        f"PASS: {pass_count}"
    )

    print(
        f"FAILED: {failed_count}"
    )

    print(
        f"INVALID: {invalid_count}"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    return results