# -*- coding: utf-8 -*-
"""
openmm_charmm_scoring.py  (v1.2.0)

OpenMM + CHARMM scoring module for the AF3 -> HADDOCK pipeline.

Purpose
-------
Evaluate already-selected protein-protein poses with an OpenMM system
parameterized by a CHARMM36-family force field. Single-point potential energy
and (optionally) a short local minimization. No MD trajectory is produced.

This module does NOT redock, reinterpret AF3 evidence, select candidates or
modify HADDOCK scores.

Two parameterization routes
---------------------------
1. PSF route (``psf_path`` present in the candidate record)
       CharmmPsfFile(psf) + CharmmParameterSet(rtf, prm)  ->  createSystem
   Uses the CHARMM topology/parameter files supplied by the caller
   (e.g. top_all36_prot.rtf + par_all36m_prot.prm).
   The coordinates are read from ``psf_pdb_path`` (the PDB written together
   with the PSF) when available, otherwise from ``pdb_path``; the atom count
   must match the PSF.

2. ForceField route (no ``psf_path``)
       PDBFile -> Modeller.addHydrogens -> ForceField("charmm36.xml")
   Uses the CHARMM36 XML that ships with OpenMM. rtf/prm files are NOT used
   in this route. Missing heavy atoms (e.g. OXT) are completed with PDBFixer
   when it is installed. ``method="auto"`` picks route 1 when a PSF is given, else
   route 2.

v1.0.0 called ``CharmmParameterSet.createSystem()``, which does not exist
(AttributeError for every sample). The parameter set only carries force-field
parameters; a system needs a PSF (route 1) or an OpenMM ForceField (route 2).

Output policy
-------------
``run_openmm_charmm_batch`` returns result dicts. When ``results_dir`` is
given it additionally writes one ``<prediction_id>.json`` per sample (used for
``skip_completed``) and ``openmm_charmm_results.csv``.

Trajectory output (v1.2.0)
--------------------------
``trajectory_dir`` -> ``<prediction_id>.dcd`` and ``pdb_dir`` ->
``<prediction_id>_minimized.pdb``. The DCD holds, in order:

    frame 0   initial coordinates
    frame 1   after local energy minimization (only if ``minimize``)
    frame 2+  one frame every ``report_interval`` steps of ``md_steps``
              steps of Langevin MD at ``temperature_kelvin`` (1 fs step)

A DCD has no topology. The companion PDB uses exactly the atoms of the
simulated system (hydrogens added by the ForceField route included), so load
the two together, e.g. ``mdtraj.load_dcd(dcd, top=pdb)``.
``md_steps=0`` (default) writes the 2-frame minimization DCD only; this is
a short vacuum (NoCutoff) calculation, not a production simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import csv
import json
import re
import time

MODULE_VERSION = "1.2.0"


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class OpenMMCharmmResult:
    prediction_id: str
    fold_id: str
    input_pdb: str
    status: str
    platform: str
    forcefield: str
    initial_potential_kj_mol: Optional[float]
    minimized_potential_kj_mol: Optional[float]
    energy_change_kj_mol: Optional[float]
    n_atoms: Optional[int]
    n_residues: Optional[int]
    elapsed_seconds: float
    error: str = ""
    method: str = ""
    psf_path: str = ""
    dcd_path: str = ""
    topology_pdb: str = ""
    md_steps: int = 0
    n_frames: int = 0
    final_potential_kj_mol: Optional[float] = None


# ---------------------------------------------------------------------------
# Optional dependency loading
# ---------------------------------------------------------------------------

def _load_openmm():
    try:
        import openmm
        from openmm import app, unit
        return openmm, app, unit
    except ImportError as exc:
        raise ImportError(
            "OpenMM is not installed in the current environment. "
            "Install/enable OpenMM before running this module."
        ) from exc


# ---------------------------------------------------------------------------
# Candidate normalization
# ---------------------------------------------------------------------------

def _candidate_value(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_candidate(record: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise TypeError("Each candidate must be a mapping/dict-like object.")

    prediction_id = str(
        _candidate_value(record, ("prediction_id", "id", "sample_id"))
        or "UNKNOWN"
    )
    fold_id = str(_candidate_value(record, ("fold_id", "fold")) or "")

    pdb_path = _candidate_value(
        record,
        ("pdb_path", "input_pdb", "haddock_ready_pdb", "structure_pdb"),
    )
    if pdb_path is None:
        raise ValueError(
            f"No PDB path found for candidate {prediction_id}. "
            "Expected one of: pdb_path, input_pdb, "
            "haddock_ready_pdb, structure_pdb."
        )

    return {
        "prediction_id": prediction_id,
        "fold_id": fold_id,
        "pdb_path": str(pdb_path),
        "psf_path": str(_candidate_value(record, ("psf_path",)) or ""),
        "psf_pdb_path": str(
            _candidate_value(record, ("psf_pdb_path",)) or ""
        ),
    }


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

def _platform_order(openmm_module, preferred: str) -> list[str]:
    """Registered platforms, preferred first, then CUDA/OpenCL/CPU/Reference."""
    registered = {
        openmm_module.Platform.getPlatform(i).getName().upper():
        openmm_module.Platform.getPlatform(i).getName()
        for i in range(openmm_module.Platform.getNumPlatforms())
    }
    order = [preferred.upper(), "CUDA", "OPENCL", "CPU", "REFERENCE"]
    seen, result = set(), []
    for key in order:
        if key in registered and key not in seen:
            seen.add(key)
            result.append(registered[key])
    return result


def create_context(openmm_module, system, integrator_factory, preferred="CUDA"):
    """
    Create a Context on the first platform that actually works.

    A platform can be registered yet unusable (e.g. OpenCL without a device),
    so availability is decided by successful Context creation, not by
    Platform.getPlatformByName().
    """
    errors = []
    for name in _platform_order(openmm_module, preferred):
        platform = openmm_module.Platform.getPlatformByName(name)
        props = {"Precision": "mixed"} if name in ("CUDA", "OpenCL") else {}
        try:
            ctx = openmm_module.Context(
                system, integrator_factory(), platform, props
            )
            return ctx, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        "No usable OpenMM platform. Tried -> " + " | ".join(errors)
    )


# ---------------------------------------------------------------------------
# CHARMM system construction
# ---------------------------------------------------------------------------

def _check_files(paths: Sequence[str | Path], label: str) -> list[str]:
    out = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(f"CHARMM {label} file not found: {p}")
        out.append(str(p))
    return out


def build_charmm_system(
    pdb_path: str | Path,
    *,
    topology_files: Sequence[str | Path] = (),
    parameter_files: Sequence[str | Path] = (),
    psf_path: str | Path | None = None,
    psf_pdb_path: str | Path | None = None,
    method: str = "auto",
    forcefield_xml: str = "charmm36.xml",
    remove_cmmotion: bool = False,
):
    """
    Build an OpenMM system for one pose.

    Returns
    -------
    (openmm, unit, topology, positions, system, method_used)
    """
    openmm, app, unit = _load_openmm()

    if method not in ("auto", "psf", "forcefield"):
        raise ValueError("method must be 'auto', 'psf' or 'forcefield'.")

    use_psf = (method == "psf") or (method == "auto" and psf_path)

    # ---------------- route 1: PSF + CHARMM rtf/prm ----------------
    if use_psf:
        if not psf_path:
            raise ValueError("method='psf' requires psf_path.")
        psf_path = Path(psf_path)
        if not psf_path.is_file():
            raise FileNotFoundError(f"PSF file not found: {psf_path}")

        coord_pdb = (
            Path(psf_pdb_path)
            if psf_pdb_path and Path(psf_pdb_path).is_file()
            else Path(pdb_path)
        )
        if not coord_pdb.is_file():
            raise FileNotFoundError(f"PDB file not found: {coord_pdb}")

        params = app.CharmmParameterSet(
            *_check_files(topology_files, "topology"),
            *_check_files(parameter_files, "parameter"),
        )
        psf = app.CharmmPsfFile(str(psf_path))
        pdb = app.PDBFile(str(coord_pdb))

        if len(pdb.positions) != len(psf.atom_list):
            raise ValueError(
                f"PSF/PDB atom count mismatch: PSF={len(psf.atom_list)} "
                f"PDB={len(pdb.positions)} ({coord_pdb.name})"
            )

        system = psf.createSystem(
            params,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=remove_cmmotion,
        )
        return openmm, unit, psf.topology, pdb.positions, system, "psf"

    # ---------------- route 2: OpenMM ForceField ----------------
    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    ff = app.ForceField(forcefield_xml)

    # PDBFixer completes missing heavy atoms (typically the C-terminal OXT);
    # without it OpenMM rejects chain ends that have no terminal template match.
    # Missing *residues* are never invented (missingResidues = {}).
    try:
        from pdbfixer import PDBFixer
    except ImportError:
        PDBFixer = None

    if PDBFixer is not None:
        fixer = PDBFixer(filename=str(pdb_path))
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        topology, positions = fixer.topology, fixer.positions
    else:
        pdb = app.PDBFile(str(pdb_path))
        topology, positions = pdb.topology, pdb.positions

    # Hydrogen placement uses random numbers; seed for (near-)reproducibility.
    import random
    random.seed(0)
    try:
        import numpy as np
        np.random.seed(0)
    except ImportError:
        pass

    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens(ff)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=remove_cmmotion,
    )
    return (
        openmm, unit, modeller.topology, modeller.positions, system,
        "forcefield",
    )


# ---------------------------------------------------------------------------
# Energy evaluation
# ---------------------------------------------------------------------------

def _write_frame(dcd, context, unit):
    state = context.getState(getPositions=True)
    dcd.writeModel(state.getPositions())


def evaluate_pose(
    pdb_path: str | Path,
    *,
    topology_files: Sequence[str | Path] = (),
    parameter_files: Sequence[str | Path] = (),
    psf_path: str | Path | None = None,
    psf_pdb_path: str | Path | None = None,
    method: str = "auto",
    platform: str = "CUDA",
    minimize: bool = True,
    max_iterations: int = 500,
    tolerance_kj_mol_per_nm: float = 10.0,
    dcd_path: str | Path | None = None,
    topology_pdb_path: str | Path | None = None,
    md_steps: int = 0,
    report_interval: int = 1000,
    temperature_kelvin: float = 300.0,
) -> OpenMMCharmmResult:
    """
    Single-point energy, optional minimization, optional short MD.

    ``dcd_path`` enables trajectory output (see module docstring); a companion
    PDB with the simulated atoms is written to ``topology_pdb_path`` (default:
    next to the DCD, ``<stem>_minimized.pdb``).
    """
    started = time.perf_counter()
    context = None
    dcd_file = None
    dcd_handle = None
    method_used = method
    n_frames = 0

    try:
        if md_steps < 0 or report_interval < 1:
            raise ValueError("md_steps must be >= 0 and report_interval >= 1.")

        (openmm, unit, topology, positions, system,
         method_used) = build_charmm_system(
            pdb_path,
            topology_files=topology_files,
            parameter_files=parameter_files,
            psf_path=psf_path,
            psf_pdb_path=psf_pdb_path,
            method=method,
        )
        from openmm import app

        dt = 1.0 * unit.femtoseconds
        if md_steps > 0:
            integrator_factory = lambda: openmm.LangevinMiddleIntegrator(
                temperature_kelvin * unit.kelvin,
                1.0 / unit.picosecond,
                dt,
            )
        else:
            integrator_factory = lambda: openmm.VerletIntegrator(dt)

        context, platform_name = create_context(
            openmm, system, integrator_factory, preferred=platform,
        )
        context.setPositions(positions)

        if dcd_path:
            dcd_path = Path(dcd_path)
            dcd_path.parent.mkdir(parents=True, exist_ok=True)
            dcd_handle = open(dcd_path, "wb")
            dcd_file = app.DCDFile(
                dcd_handle,
                topology,
                dt,
                0,
                report_interval,
            )
            _write_frame(dcd_file, context, unit)          # frame 0: initial
            n_frames += 1

        initial = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        minimized = initial

        if minimize:
            openmm.LocalEnergyMinimizer.minimize(
                context,
                tolerance=(
                    tolerance_kj_mol_per_nm
                    * unit.kilojoule_per_mole / unit.nanometer
                ),
                maxIterations=max_iterations,
            )
            minimized = (
                context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
            if dcd_file:
                _write_frame(dcd_file, context, unit)      # frame 1: minimized
                n_frames += 1

        # Companion topology = the minimized (or initial) atoms of this system.
        topology_pdb = ""
        if dcd_path:
            topology_pdb = str(
                Path(topology_pdb_path)
                if topology_pdb_path
                else Path(dcd_path).with_name(Path(dcd_path).stem + "_minimized.pdb")
            )
            Path(topology_pdb).parent.mkdir(parents=True, exist_ok=True)
            with open(topology_pdb, "w") as fh:
                app.PDBFile.writeFile(
                    topology,
                    context.getState(getPositions=True).getPositions(),
                    fh,
                    keepIds=True,
                )

        final = minimized
        if md_steps > 0:
            done = 0
            while done < md_steps:
                n = min(report_interval, md_steps - done)
                context.getIntegrator().step(n)
                done += n
                if dcd_file:
                    _write_frame(dcd_file, context, unit)
                    n_frames += 1
            final = (
                context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )

        return OpenMMCharmmResult(
            prediction_id="UNKNOWN",
            fold_id="",
            input_pdb=str(Path(pdb_path)),
            status="PASS",
            platform=platform_name,
            forcefield="CHARMM36" + ("m" if method_used == "psf" else ""),
            initial_potential_kj_mol=float(initial),
            minimized_potential_kj_mol=float(minimized),
            energy_change_kj_mol=float(minimized - initial),
            n_atoms=topology.getNumAtoms(),
            n_residues=topology.getNumResidues(),
            elapsed_seconds=time.perf_counter() - started,
            method=method_used,
            psf_path=str(psf_path or ""),
            dcd_path=str(dcd_path or ""),
            topology_pdb=topology_pdb,
            md_steps=int(md_steps),
            n_frames=n_frames,
            final_potential_kj_mol=float(final),
        )

    except Exception as exc:
        return OpenMMCharmmResult(
            prediction_id="UNKNOWN",
            fold_id="",
            input_pdb=str(Path(pdb_path)),
            status="FAILED",
            platform=str(platform),
            forcefield="CHARMM",
            initial_potential_kj_mol=None,
            minimized_potential_kj_mol=None,
            energy_change_kj_mol=None,
            n_atoms=None,
            n_residues=None,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            method=method_used,
            psf_path=str(psf_path or ""),
        )
    finally:
        if dcd_handle is not None:
            dcd_handle.close()   # DCDFile does not close the handle itself
        del context  # release GPU memory between samples


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _resolve_candidates(candidate_object: Any) -> list:
    if isinstance(candidate_object, Mapping):
        for key in ("records", "candidates"):
            if isinstance(candidate_object.get(key), list):
                return candidate_object[key]
        raise ValueError(
            "Candidate mapping does not contain 'records' or 'candidates'."
        )
    if isinstance(candidate_object, (list, tuple)):
        return list(candidate_object)
    for attr in ("records", "candidates"):
        if hasattr(candidate_object, attr):
            return list(getattr(candidate_object, attr))
    raise TypeError(
        "Unsupported candidate_object. Expected a list/tuple, mapping with "
        "records/candidates, or an object exposing records/candidates."
    )


def _settings(method, platform, minimize, max_iterations,
              md_steps, report_interval, temperature_kelvin, save_traj):
    return {
        "module_version": MODULE_VERSION,
        "method": method,
        "minimize": bool(minimize),
        "max_iterations": int(max_iterations),
        "md_steps": int(md_steps),
        "report_interval": int(report_interval),
        "temperature_kelvin": float(temperature_kelvin),
        "save_trajectory": bool(save_traj),
    }


def run_openmm_charmm_batch(
    candidate_object: Any,
    *,
    topology_files: Sequence[str | Path] = (),
    parameter_files: Sequence[str | Path] = (),
    method: str = "auto",
    platform: str = "CUDA",
    minimize: bool = True,
    max_iterations: int = 500,
    skip_completed: bool = True,
    results_dir: str | Path | None = None,
    trajectory_dir: str | Path | None = None,
    pdb_dir: str | Path | None = None,
    md_steps: int = 0,
    report_interval: int = 1000,
    temperature_kelvin: float = 300.0,
) -> list[dict[str, Any]]:
    """
    Run OpenMM + CHARMM evaluation on every candidate.

    ``skip_completed`` only has an effect when ``results_dir`` is given: a
    sample is skipped when ``<results_dir>/<prediction_id>.json`` exists with
    status PASS, the same input PDB / PSF and identical settings, and (when
    ``trajectory_dir`` is set) its DCD and topology PDB still exist.

    ``trajectory_dir`` enables DCD output; ``pdb_dir`` (default:
    ``trajectory_dir``) receives the companion ``<id>_minimized.pdb``.
    """
    candidates = _resolve_candidates(candidate_object)
    if not candidates:
        raise RuntimeError("No selected candidates were supplied.")

    out_dir = Path(results_dir) if results_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    wanted = _settings(
        method, platform, minimize, max_iterations,
        md_steps, report_interval, temperature_kelvin, bool(trajectory_dir),
    )
    traj_dir = Path(trajectory_dir) if trajectory_dir else None
    top_dir = Path(pdb_dir) if pdb_dir else traj_dir
    results: list[dict[str, Any]] = []

    for index, raw in enumerate(candidates, start=1):
        cand = normalize_candidate(raw)
        pid = cand["prediction_id"]
        print(f"[{index}/{len(candidates)}] {pid}")

        cache = out_dir / f"{_safe(pid)}.json" if out_dir else None

        if skip_completed and cache and cache.is_file():
            try:
                saved = json.loads(cache.read_text(encoding="utf-8"))
                r0 = saved["result"]
                files_ok = (
                    not traj_dir
                    or (Path(r0.get("dcd_path", "")).is_file()
                        and Path(r0.get("topology_pdb", "")).is_file())
                )
                if (
                    files_ok
                    and saved["result"]["status"] == "PASS"
                    and saved["settings"] == wanted
                    and saved["result"]["input_pdb"] == cand["pdb_path"]
                    and saved["result"].get("psf_path", "")
                    == cand["psf_path"]
                ):
                    results.append(saved["result"])
                    print("  status   : SKIPPED (cached PASS)")
                    continue
            except Exception:
                pass  # unreadable cache -> recompute

        result = evaluate_pose(
            cand["pdb_path"],
            topology_files=topology_files,
            parameter_files=parameter_files,
            psf_path=cand["psf_path"] or None,
            psf_pdb_path=cand["psf_pdb_path"] or None,
            method=method,
            platform=platform,
            minimize=minimize,
            max_iterations=max_iterations,
            dcd_path=(traj_dir / f"{_safe(pid)}.dcd") if traj_dir else None,
            topology_pdb_path=(
                top_dir / f"{_safe(pid)}_minimized.pdb" if traj_dir else None
            ),
            md_steps=md_steps,
            report_interval=report_interval,
            temperature_kelvin=temperature_kelvin,
        )
        result.prediction_id = pid
        result.fold_id = cand["fold_id"]

        row = asdict(result)
        results.append(row)

        print(f"  status   : {result.status}")
        print(f"  platform : {result.platform} | method: {result.method}")
        if result.status == "PASS":
            print(f"  initial  : {result.initial_potential_kj_mol:.3f} kJ/mol")
            print(f"  minimized: {result.minimized_potential_kj_mol:.3f} kJ/mol")
            if result.dcd_path:
                print(f"  dcd      : {result.dcd_path} ({result.n_frames} frames)")
        else:
            print(f"  error    : {result.error}")

        if cache:
            cache.write_text(
                json.dumps({"settings": wanted, "result": row}, indent=2),
                encoding="utf-8",
            )

    if out_dir and results:
        csv_path = out_dir / "openmm_charmm_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)

    return results


def summarize_openmm(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    summary = {"total": len(results), "pass": 0, "failed": 0}
    for r in results:
        summary["pass" if r.get("status") == "PASS" else "failed"] += 1
    return summary


if __name__ == "__main__":
    print(f"openmm_charmm_scoring.py {MODULE_VERSION}")
