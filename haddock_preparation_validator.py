#!/usr/bin/env python3

"""
haddock_preparation_validator.py

AF3-derived PDB -> HADDOCK preparation / validation layer.

Responsibilities
----------------
1. Discover AF3-derived PDB files.
2. Validate PDB readability.
3. Validate expected protein chains.
4. Validate atom / residue counts.
5. Inspect residue numbering and insertion codes.
6. Inspect HETATM / non-standard residues.
7. Create HADDOCK-ready PDB copies.
8. Generate validation JSON and manifest.

Explicitly NOT handled here
----------------------------
- HADDOCK execution
- HADDOCK configuration generation
- AIR generation
- AF3 interface confidence classification
- Docking
- MD / OpenMM

Those functions belong to downstream modules.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import gemmi
except ImportError:
    gemmi = None


# ============================================================
# Standard protein residues
# ============================================================

STANDARD_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


# ============================================================
# Data structures
# ============================================================

@dataclass
class ChainValidation:

    chain_id: str

    residue_count: int

    atom_count: int

    first_residue_number: Optional[int]

    last_residue_number: Optional[int]

    insertion_codes: List[str]

    duplicate_residue_numbers: List[str]

    nonstandard_residues: List[str]


@dataclass
class PreparationResult:

    input_pdb: str

    output_pdb: str

    status: str

    chains: List[str]

    residue_count: int

    atom_count: int

    hetatm_count: int

    nonstandard_residues: List[str]

    duplicate_residue_numbers: List[str]

    insertion_codes: List[str]

    chain_validation: List[Dict[str, Any]]

    warnings: List[str]

    errors: List[str]

    message: str


# ============================================================
# Dependency
# ============================================================

def check_dependencies() -> None:

    if gemmi is None:

        raise ImportError(
            "\n"
            "Gemmi is required for PDB validation.\n\n"
            "Install it in Colab with:\n\n"
            "!pip install gemmi\n"
        )


# ============================================================
# Load PDB
# ============================================================

def load_pdb(
    pdb_path: str | Path
):

    check_dependencies()

    pdb_path = Path(pdb_path)

    if not pdb_path.exists():

        raise FileNotFoundError(
            f"PDB file does not exist:\n{pdb_path}"
        )

    if pdb_path.stat().st_size == 0:

        raise ValueError(
            f"PDB file is empty:\n{pdb_path}"
        )

    try:

        structure = gemmi.read_structure(
            str(pdb_path)
        )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to parse PDB:\n"
            f"{pdb_path}\n\n"
            f"Error: {exc}"
        ) from exc

    return structure


# ============================================================
# Chain extraction
# ============================================================

def get_chain_ids(
    structure
) -> List[str]:

    chain_ids = []

    for model in structure:

        for chain in model:

            chain_id = str(
                chain.name
            )

            if chain_id not in chain_ids:

                chain_ids.append(
                    chain_id
                )

    return sorted(
        chain_ids
    )


# ============================================================
# Residue key
# ============================================================

def residue_key(
    residue
) -> str:

    seqid = residue.seqid

    number = seqid.num

    insertion = str(
        seqid.icode
    ).strip()

    if insertion:

        return f"{number}{insertion}"

    return str(number)


# ============================================================
# Chain validation
# ============================================================

def validate_chain(
    chain
) -> ChainValidation:

    residue_count = 0
    atom_count = 0

    residue_numbers = []
    insertion_codes = []

    nonstandard_residues = []

    for residue in chain:

        residue_count += 1

        atom_count += len(
            residue
        )

        key = residue_key(
            residue
        )

        residue_numbers.append(
            key
        )

        insertion = str(
            residue.seqid.icode
        ).strip()

        if insertion:

            insertion_codes.append(
                insertion
            )

        residue_name = str(
            residue.name
        ).upper().strip()

        if residue_name not in STANDARD_PROTEIN_RESIDUES:

            if residue_name not in nonstandard_residues:

                nonstandard_residues.append(
                    residue_name
                )

    duplicate_residue_numbers = []

    seen = set()

    for key in residue_numbers:

        if key in seen:

            if key not in duplicate_residue_numbers:

                duplicate_residue_numbers.append(
                    key
                )

        seen.add(
            key
        )

    first_residue_number = None
    last_residue_number = None

    if len(residue_numbers) > 0:

        try:

            first_residue_number = int(
                re.match(
                    r"-?\d+",
                    residue_numbers[0]
                ).group()
            )

            last_residue_number = int(
                re.match(
                    r"-?\d+",
                    residue_numbers[-1]
                ).group()
            )

        except Exception:

            first_residue_number = None
            last_residue_number = None

    return ChainValidation(
        chain_id=str(
            chain.name
        ),
        residue_count=residue_count,
        atom_count=atom_count,
        first_residue_number=first_residue_number,
        last_residue_number=last_residue_number,
        insertion_codes=sorted(
            set(insertion_codes)
        ),
        duplicate_residue_numbers=sorted(
            set(duplicate_residue_numbers)
        ),
        nonstandard_residues=sorted(
            set(nonstandard_residues)
        ),
    )


# ============================================================
# Whole-structure validation
# ============================================================

def validate_structure(
    pdb_path: str | Path,
    expected_chains: Optional[List[str]] = None
) -> Dict[str, Any]:

    structure = load_pdb(
        pdb_path
    )

    chain_ids = get_chain_ids(
        structure
    )

    residue_count = 0
    atom_count = 0
    hetatm_count = 0

    chain_results = []

    warnings = []
    errors = []

    # --------------------------------------------------------
    # Chain validation
    # --------------------------------------------------------

    for model in structure:

        for chain in model:

            result = validate_chain(
                chain
            )

            chain_results.append(
                result
            )

            residue_count += (
                result.residue_count
            )

            atom_count += (
                result.atom_count
            )

            if result.nonstandard_residues:

                warnings.append(
                    f"Chain {result.chain_id} "
                    f"contains non-standard residues: "
                    f"{result.nonstandard_residues}"
                )

            if result.insertion_codes:

                warnings.append(
                    f"Chain {result.chain_id} "
                    f"contains insertion codes: "
                    f"{result.insertion_codes}"
                )

            if result.duplicate_residue_numbers:

                errors.append(
                    f"Chain {result.chain_id} "
                    f"contains duplicate residue numbers: "
                    f"{result.duplicate_residue_numbers}"
                )

    # --------------------------------------------------------
    # HETATM inspection
    # --------------------------------------------------------

    for model in structure:

        for chain in model:

            for residue in chain:

                for atom in residue:

                    # Gemmi stores standard protein atoms normally.
                    # Hetero flag is available through residue het_flag.
                    if getattr(
                        residue,
                        "het_flag",
                        "A"
                    ) not in ("A", ""):

                        hetatm_count += 1

    # --------------------------------------------------------
    # Expected chain validation
    # --------------------------------------------------------

    if expected_chains is not None:

        expected = sorted(
            expected_chains
        )

        observed = sorted(
            chain_ids
        )

        if observed != expected:

            errors.append(
                "Unexpected chain set. "
                f"Expected={expected}; "
                f"Observed={observed}"
            )

    # --------------------------------------------------------
    # Empty structure
    # --------------------------------------------------------

    if atom_count == 0:

        errors.append(
            "No atoms were found in the PDB."
        )

    if residue_count == 0:

        errors.append(
            "No residues were found in the PDB."
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if errors:

        status = "INVALID"

    elif warnings:

        status = "PASS_WITH_WARNINGS"

    else:

        status = "PASS"

    all_nonstandard = sorted(
        {
            residue
            for result in chain_results
            for residue in result.nonstandard_residues
        }
    )

    all_duplicate_numbers = sorted(
        {
            residue
            for result in chain_results
            for residue in result.duplicate_residue_numbers
        }
    )

    all_insertion_codes = sorted(
        {
            code
            for result in chain_results
            for code in result.insertion_codes
        }
    )

    return {
        "status": status,
        "chains": chain_ids,
        "residue_count": residue_count,
        "atom_count": atom_count,
        "hetatm_count": hetatm_count,
        "nonstandard_residues": all_nonstandard,
        "duplicate_residue_numbers": all_duplicate_numbers,
        "insertion_codes": all_insertion_codes,
        "chain_validation": [
            asdict(result)
            for result in chain_results
        ],
        "warnings": warnings,
        "errors": errors,
    }


# ============================================================
# HADDOCK-ready PDB preparation
# ============================================================

def prepare_haddock_pdb(
    input_pdb: str | Path,
    output_pdb: str | Path
) -> Path:

    """
    Create a HADDOCK preparation copy.

    Important:
    This function intentionally does NOT perform aggressive
    structural modifications.

    HADDOCK itself has its own preprocessing layer.

    Therefore this function currently:
    - loads the validated PDB
    - writes a clean PDB copy
    - preserves the molecular structure
    """

    check_dependencies()

    input_pdb = Path(
        input_pdb
    )

    output_pdb = Path(
        output_pdb
    )

    structure = load_pdb(
        input_pdb
    )

    output_pdb.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    structure.write_pdb(
        str(output_pdb)
    )

    if not output_pdb.exists():

        raise RuntimeError(
            "Failed to generate HADDOCK-ready PDB:\n"
            f"{output_pdb}"
        )

    if output_pdb.stat().st_size == 0:

        raise RuntimeError(
            "Generated HADDOCK-ready PDB is empty:\n"
            f"{output_pdb}"
        )

    return output_pdb


# ============================================================
# Single PDB processing
# ============================================================

def process_haddock_preparation(
    input_pdb: str | Path,
    output_directory: str | Path,
    expected_chains: Optional[List[str]] = None
) -> PreparationResult:

    input_pdb = Path(
        input_pdb
    )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    try:

        validation = validate_structure(
            input_pdb,
            expected_chains=expected_chains
        )

    except Exception as exc:

        return PreparationResult(
            input_pdb=str(input_pdb),
            output_pdb="",
            status="FAILED",
            chains=[],
            residue_count=0,
            atom_count=0,
            hetatm_count=0,
            nonstandard_residues=[],
            duplicate_residue_numbers=[],
            insertion_codes=[],
            chain_validation=[],
            warnings=[],
            errors=[str(exc)],
            message="PDB validation failed."
        )

    # --------------------------------------------------------
    # Do not prepare invalid structures
    # --------------------------------------------------------

    if validation["status"] == "INVALID":

        return PreparationResult(
            input_pdb=str(input_pdb),
            output_pdb="",
            status="INVALID",
            chains=validation["chains"],
            residue_count=validation["residue_count"],
            atom_count=validation["atom_count"],
            hetatm_count=validation["hetatm_count"],
            nonstandard_residues=validation[
                "nonstandard_residues"
            ],
            duplicate_residue_numbers=validation[
                "duplicate_residue_numbers"
            ],
            insertion_codes=validation[
                "insertion_codes"
            ],
            chain_validation=validation[
                "chain_validation"
            ],
            warnings=validation[
                "warnings"
            ],
            errors=validation[
                "errors"
            ],
            message=(
                "PDB validation failed. "
                "HADDOCK-ready PDB was not generated."
            )
        )

    # --------------------------------------------------------
    # Output name
    # --------------------------------------------------------

    output_pdb = (
        output_directory /
        f"{input_pdb.stem}_haddock_ready.pdb"
    )

    # --------------------------------------------------------
    # Create prepared PDB
    # --------------------------------------------------------

    try:

        prepare_haddock_pdb(
            input_pdb=input_pdb,
            output_pdb=output_pdb
        )

    except Exception as exc:

        return PreparationResult(
            input_pdb=str(input_pdb),
            output_pdb=str(output_pdb),
            status="FAILED",
            chains=validation["chains"],
            residue_count=validation["residue_count"],
            atom_count=validation["atom_count"],
            hetatm_count=validation["hetatm_count"],
            nonstandard_residues=validation[
                "nonstandard_residues"
            ],
            duplicate_residue_numbers=validation[
                "duplicate_residue_numbers"
            ],
            insertion_codes=validation[
                "insertion_codes"
            ],
            chain_validation=validation[
                "chain_validation"
            ],
            warnings=validation[
                "warnings"
            ],
            errors=[
                f"Preparation failed: {exc}"
            ],
            message="Failed to create HADDOCK-ready PDB."
        )

    # --------------------------------------------------------
    # Revalidate generated PDB
    # --------------------------------------------------------

    try:

        prepared_validation = validate_structure(
            output_pdb,
            expected_chains=expected_chains
        )

    except Exception as exc:

        return PreparationResult(
            input_pdb=str(input_pdb),
            output_pdb=str(output_pdb),
            status="FAILED",
            chains=[],
            residue_count=0,
            atom_count=0,
            hetatm_count=0,
            nonstandard_residues=[],
            duplicate_residue_numbers=[],
            insertion_codes=[],
            chain_validation=[],
            warnings=[],
            errors=[
                f"Prepared PDB revalidation failed: {exc}"
            ],
            message=(
                "Generated PDB could not be revalidated."
            )
        )

    if prepared_validation["status"] == "INVALID":

        return PreparationResult(
            input_pdb=str(input_pdb),
            output_pdb=str(output_pdb),
            status="INVALID",
            chains=prepared_validation["chains"],
            residue_count=prepared_validation[
                "residue_count"
            ],
            atom_count=prepared_validation[
                "atom_count"
            ],
            hetatm_count=prepared_validation[
                "hetatm_count"
            ],
            nonstandard_residues=prepared_validation[
                "nonstandard_residues"
            ],
            duplicate_residue_numbers=prepared_validation[
                "duplicate_residue_numbers"
            ],
            insertion_codes=prepared_validation[
                "insertion_codes"
            ],
            chain_validation=prepared_validation[
                "chain_validation"
            ],
            warnings=prepared_validation[
                "warnings"
            ],
            errors=prepared_validation[
                "errors"
            ],
            message=(
                "Prepared PDB failed revalidation."
            )
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    result = PreparationResult(
        input_pdb=str(input_pdb),
        output_pdb=str(output_pdb),
        status=prepared_validation["status"],
        chains=prepared_validation["chains"],
        residue_count=prepared_validation[
            "residue_count"
        ],
        atom_count=prepared_validation[
            "atom_count"
        ],
        hetatm_count=prepared_validation[
            "hetatm_count"
        ],
        nonstandard_residues=prepared_validation[
            "nonstandard_residues"
        ],
        duplicate_residue_numbers=prepared_validation[
            "duplicate_residue_numbers"
        ],
        insertion_codes=prepared_validation[
            "insertion_codes"
        ],
        chain_validation=prepared_validation[
            "chain_validation"
        ],
        warnings=prepared_validation[
            "warnings"
        ],
        errors=prepared_validation[
            "errors"
        ],
        message=(
            "PDB validated and HADDOCK-ready copy "
            "generated successfully."
        )
    )

    manifest_path = (
        output_directory /
        "haddock_preparation_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            asdict(result),
            f,
            indent=2,
            ensure_ascii=False
        )

    return result


# ============================================================
# Batch processing
# ============================================================

def process_haddock_preparation_directory(
    input_directory: str | Path,
    output_directory: str | Path,
    expected_chains: Optional[List[str]] = None
) -> List[PreparationResult]:

    """
    Process all PDB files in an input directory.

    Intended for later batch processing of fold_NNN outputs.
    """

    input_directory = Path(
        input_directory
    )

    output_directory = Path(
        output_directory
    )

    if not input_directory.exists():

        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{input_directory}"
        )

    pdb_files = sorted(
        input_directory.glob(
            "*.pdb"
        )
    )

    if len(pdb_files) == 0:

        raise FileNotFoundError(
            f"No PDB files found in:\n"
            f"{input_directory}"
        )

    results = []

    for pdb_path in pdb_files:

        print(
            "\n" + "=" * 70
        )

        print(
            f"Processing: {pdb_path.name}"
        )

        result = process_haddock_preparation(
            input_pdb=pdb_path,
            output_directory=output_directory,
            expected_chains=expected_chains
        )

        results.append(
            result
        )

        print(
            f"Status: {result.status}"
        )

        print(
            f"Chains: {result.chains}"
        )

        print(
            f"Residues: {result.residue_count}"
        )

        print(
            f"Atoms: {result.atom_count}"
        )

        if result.warnings:

            print(
                f"Warnings: {len(result.warnings)}"
            )

        if result.errors:

            print(
                f"Errors: {len(result.errors)}"
            )

        if result.output_pdb:

            print(
                f"Output: {result.output_pdb}"
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

    batch_manifest_path = (
        output_directory /
        "haddock_preparation_batch_manifest.json"
    )

    with open(
        batch_manifest_path,
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
        result.status == "PASS"
        for result in results
    )

    warning_count = sum(
        result.status == "PASS_WITH_WARNINGS"
        for result in results
    )

    invalid_count = sum(
        result.status == "INVALID"
        for result in results
    )

    failed_count = sum(
        result.status == "FAILED"
        for result in results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "HADDOCK PREPARATION COMPLETE"
    )

    print(
        f"Total: {len(results)}"
    )

    print(
        f"PASS: {pass_count}"
    )

    print(
        f"PASS_WITH_WARNINGS: {warning_count}"
    )

    print(
        f"INVALID: {invalid_count}"
    )

    print(
        f"FAILED: {failed_count}"
    )

    print(
        f"Manifest: {batch_manifest_path}"
    )

    return results