# -*- coding: utf-8 -*-
"""
trajectory_viewer.py  (v1.4.0)

Analysis and 3D visualization of the DCD trajectories written by
``openmm_charmm_scoring.py`` (v1.2.0+) for the AF3 -> HADDOCK -> OpenMM pipeline.

Packages and their roles
------------------------
mdtraj      load DCD+PDB, superposition, RMSD, Rg, SASA/BSA, multi-model PDB
MDAnalysis  per-residue RMSF (AlignTraj + RMSF) and an independent
            interface-residue selection used as a cross-check
py3Dmol     interactive animation / static views / side-by-side grid

Input layout (written by the scoring module)
--------------------------------------------
    <results_dir>/<prediction_id>.json           <- status + settings (contract)
    <trajectory_dir>/<prediction_id>.dcd
    <pdb_dir>/<prediction_id>_minimized.pdb      <- topology of the DCD

Use :func:`discover_from_results` (v1.1.0). It reads the per-sample result
JSON of the scoring module, so only status PASS trajectories are used (a
failed MD run can leave a partial DCD behind) and the frame layout
(``minimize``, ``report_interval``) comes from the run itself instead of a
setting that has to be kept in sync by hand. :func:`discover_trajectories`
(directory scan) is kept as a fallback for DCDs without result JSONs.

Sample selection (v1.4.0): :func:`select_kneedle_and` sets one Kneedle elbow
cutoff per score column (same algorithm as the evidence cell of the pipeline,
``evidence_cutoff.py``) and keeps the samples that pass every cutoff (AND).

DCD frame convention of the scoring module:
    frame 0 = initial, frame 1 = minimized, frame 2+ = MD (every
    ``report_interval`` steps, 1 fs). With ``md_steps=0`` there are 2 frames.
Use ``minimized=False`` in the functions below if minimization was disabled.

Two-chain complexes (chains taken in file order, normally A and B) are assumed
for the complex metrics (interface, buried surface area, COM distance).
All metrics use heavy atoms only (hydrogens added by OpenMM are dropped).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence
import tempfile
import warnings

import numpy as np
import pandas as pd

MODULE_VERSION = "1.4.0"

CHAIN_COLORS = ["#4C78A8", "#E45756", "#54A24B", "#F58518"]


# ---------------------------------------------------------------------------
# Lazy imports (give an actionable message in Colab)
# ---------------------------------------------------------------------------

def _mdtraj():
    try:
        import mdtraj
        return mdtraj
    except ImportError as exc:
        raise ImportError("mdtraj is missing: pip install mdtraj") from exc


def _mda():
    try:
        import MDAnalysis
        return MDAnalysis
    except ImportError as exc:
        raise ImportError("MDAnalysis is missing: pip install mdanalysis") from exc


def _evidence_cutoff():
    try:
        import evidence_cutoff
        return evidence_cutoff
    except ImportError as exc:
        raise ImportError(
            "evidence_cutoff.py (v1.0.0+) is missing: put it in the same "
            "modules folder as trajectory_viewer.py"
        ) from exc


def _py3dmol():
    try:
        import py3Dmol
        return py3Dmol
    except ImportError as exc:
        raise ImportError("py3Dmol is missing: pip install py3Dmol") from exc


# ---------------------------------------------------------------------------
# Discovery / selection
# ---------------------------------------------------------------------------

def discover_trajectories(
    trajectory_dir: str | Path,
    pdb_dir: str | Path | None = None,
    merged_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fallback: pair every ``<id>.dcd`` with its ``<id>_minimized.pdb`` by
    scanning the folder. Prefer :func:`discover_from_results`, which skips
    failed runs and knows each run's frame layout.

    When ``merged_csv`` (the EMscoring + OpenMM table of the pipeline notebook)
    exists, score columns are joined on ``prediction_id``.
    """
    trajectory_dir = Path(trajectory_dir)
    pdb_dir = Path(pdb_dir) if pdb_dir else trajectory_dir
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

    rows = []
    for dcd in sorted(trajectory_dir.glob("*.dcd")):
        pid = dcd.stem
        top = pdb_dir / f"{pid}_minimized.pdb"
        rows.append({
            "prediction_id": pid,
            "fold_id": pid[:8],
            "dcd_path": str(dcd),
            "topology_pdb": str(top),
            "has_topology": top.is_file(),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return _join_scores(df, merged_csv)


def discover_from_results(
    results_dir: str | Path,
    merged_csv: str | Path | None = None,
    trajectory_dir: str | Path | None = None,
    pdb_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build the trajectory table from ``<results_dir>/<prediction_id>.json``
    written by ``openmm_charmm_scoring.run_openmm_charmm_batch``.

    Every JSON becomes one row; ``usable`` is True only when the run status is
    PASS, trajectory output was enabled and both files exist. ``reason`` says
    why a row is not usable. ``minimized`` / ``report_interval`` / ``md_steps``
    are taken from the run settings.

    ``trajectory_dir`` / ``pdb_dir``: fallback locations used when the stored
    absolute paths do not exist (e.g. Drive mounted at another path).
    """
    import json

    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result, settings = data["result"], data["settings"]
        except Exception:
            continue                      # not a per-sample result file

        pid = str(result.get("prediction_id") or path.stem)
        dcd = Path(result.get("dcd_path") or "")
        top = Path(result.get("topology_pdb") or "")
        if trajectory_dir and not dcd.is_file():
            dcd = Path(trajectory_dir) / f"{path.stem}.dcd"
        if pdb_dir and not top.is_file():
            top = Path(pdb_dir) / f"{path.stem}_minimized.pdb"

        if result.get("status") != "PASS":
            reason = f"status {result.get('status')}: {str(result.get('error', ''))[:80]}"
        elif not settings.get("save_trajectory", bool(result.get("dcd_path"))):
            reason = "trajectory output was disabled for this run"
        elif not dcd.is_file():
            reason = f"DCD missing: {dcd.name or '-'}"
        elif not top.is_file():
            reason = f"topology PDB missing: {top.name or '-'}"
        else:
            reason = ""

        rows.append({
            "prediction_id": pid,
            "fold_id": result.get("fold_id") or pid[:8],
            "dcd_path": str(dcd),
            "topology_pdb": str(top),
            "has_topology": top.is_file(),
            "usable": reason == "",
            "reason": reason,
            "status": result.get("status"),
            "minimized": bool(settings.get("minimize", True)),
            "report_interval": settings.get("report_interval"),
            "md_steps": settings.get("md_steps", 0),
            "n_frames_expected": result.get("n_frames"),
            "method": result.get("method"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return _join_scores(df, merged_csv)


def _join_scores(df: pd.DataFrame, merged_csv) -> pd.DataFrame:
    if merged_csv and Path(merged_csv).is_file():
        scores = pd.read_csv(merged_csv)
        keep = [c for c in (
            "prediction_id", "haddock_score", "evdw", "eelec", "edesolv",
            "bsa", "initial_potential_kj_mol", "minimized_potential_kj_mol",
        ) if c in scores.columns and (c == "prediction_id" or c not in df.columns)]
        scores = scores[keep].rename(columns={"bsa": "em_bsa"})
        df = df.merge(scores, on="prediction_id", how="left")
    return df


def select_kneedle_and(
    table: pd.DataFrame,
    columns: Sequence[str],
    direction: str = "lower",
    transform: Optional[dict[str, Optional[str]]] = None,
    min_points: int = 10,
    min_distance: float = 0.05,
    bootstrap: int = 200,
    order_by: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Kneedle cutoff per column, then AND.

    For every column independently (on all rows that have a value):
      sort best -> worst (``direction="lower"``: smaller is better), find the
      elbow with the Kneedle algorithm (``evidence_cutoff.select_better_side``),
      and mark the rows at least as good as the elbow value. Without an elbow
      the 95th-percentile gap rule is used (``method_used = gap95_fallback``).
    A row is selected only if it passes the cutoff of EVERY column; a missing
    value counts as not passing.

    Returns
    -------
    selected : passing rows ordered by ``order_by`` (default: last column),
               with ``rank`` and ``<column>_rank`` (rank among all rows with a
               value, 1 = best).
    report   : one row per column: method_used, shape, cutoff_value,
               n_available, n_missing, n_pass and bootstrap percentiles of the
               cutoff (``bootstrap=0`` to skip).
    passes   : boolean table (prediction_id + one column per score) for every
               row of ``table``.
    """
    ec = _evidence_cutoff()
    columns = list(columns)
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise KeyError(f"Columns not in table: {missing}. Available: {list(table.columns)}")
    transform = transform or {}

    passes = pd.DataFrame({"prediction_id": table["prediction_id"].values}, index=table.index)
    ranks = {}
    report = []
    for col in columns:
        valid = table[col].notna()
        vals = table.loc[valid, col].astype(float).to_numpy()
        res = ec.select_better_side(
            vals, direction, method="kneedle", transform=transform.get(col),
            min_points=min_points, min_distance=min_distance,
        )
        ok = pd.Series(False, index=table.index)
        ok.loc[valid] = res["mask"]
        passes[col] = ok.values
        ranks[col] = table[col].rank(method="min", ascending=(direction == "lower"))

        row = {
            "column": col,
            "method_used": res["method_used"],
            "shape": res.get("shape"),
            "cutoff_value": res["cutoff_value"],
            "n_available": int(valid.sum()),
            "n_missing": int((~valid).sum()),
            "n_pass": int(ok.sum()),
        }
        if bootstrap and res["method_used"] == "kneedle":
            b = ec.bootstrap_cutoff(vals, direction, transform.get(col), n_boot=bootstrap,
                                    min_points=min_points, min_distance=min_distance)
            row.update(cutoff_p05=b["cutoff_p05"], cutoff_p50=b["cutoff_p50"],
                       cutoff_p95=b["cutoff_p95"], no_elbow_fraction=b["no_elbow_fraction"])
        report.append(row)

    all_pass = passes[columns].all(axis=1)
    selected = table[all_pass].copy()
    for col in columns:
        selected[f"{col}_rank"] = ranks[col][all_pass].astype(int)
    key = order_by or columns[-1]
    selected = selected.sort_values(key, ascending=(direction == "lower"), kind="mergesort")
    selected = selected.reset_index(drop=True)
    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected, pd.DataFrame(report), passes.reset_index(drop=True)


def plot_cutoffs(table: pd.DataFrame, report: pd.DataFrame, direction: str = "lower"):
    """Sorted-value curve (best -> worst) of every column with its cutoff."""
    import matplotlib.pyplot as plt

    n = len(report)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.6))
    for ax, r in zip(np.atleast_1d(axes), report.to_dict("records")):
        vals = np.sort(table[r["column"]].dropna().astype(float).to_numpy())
        if direction != "lower":
            vals = vals[::-1]
        ax.plot(np.arange(len(vals)), vals, ".-", ms=3, lw=0.8)
        if r["cutoff_value"] is not None:
            ax.axhline(r["cutoff_value"], color="tab:red", ls="--", lw=1)
            ax.axvline(r["n_pass"] - 0.5, color="tab:red", ls=":", lw=1)
        ax.set_title(f"{r['column']}\n{r['method_used']}: pass {r['n_pass']}/{r['n_available']}",
                     fontsize=9)
        ax.set_xlabel("rank (best -> worst)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def select_funnel(
    table: pd.DataFrame,
    stages: Sequence[tuple[str, int]],
    ascending: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sequential (funnel) selection.

    Each stage ``(column, n)`` sorts the rows that survived the previous stage
    by ``column`` alone and keeps the ``n`` smallest (``ascending=True``).
    Earlier columns are not used again in later stages, e.g.

        [("initial_potential_kj_mol", 50), ("haddock_score", 30),
         ("minimized_potential_kj_mol", 10)]

    keeps the 50 lowest initial potentials, then the 30 lowest HADDOCK scores
    among those 50, then the 10 lowest minimized potentials among those 30.
    Rows with a missing value in a stage's column are dropped at that stage
    (counted in the report).

    Returns
    -------
    selected : final rows, ordered by the last stage, with ``rank`` (1-based)
               and ``stage<k>_rank`` columns (rank inside each stage).
    report   : one row per stage: column, n_requested, n_in, n_missing, n_out.
    """
    current = table.copy()
    report = []
    for k, (col, n) in enumerate(stages, start=1):
        if col not in current.columns:
            raise KeyError(
                f"Stage {k}: column {col!r} not in table. "
                f"Available: {list(current.columns)}"
            )
        n_in = len(current)
        valid = current[current[col].notna()]
        current = (
            valid.sort_values(col, ascending=ascending, kind="mergesort")
            .head(int(n))
            .reset_index(drop=True)
        )
        current[f"stage{k}_rank"] = range(1, len(current) + 1)
        report.append({
            "stage": k, "column": col, "n_requested": int(n),
            "n_in": n_in, "n_missing": n_in - len(valid), "n_out": len(current),
        })
    current.insert(0, "rank", range(1, len(current) + 1))
    return current, pd.DataFrame(report)


def select_by_keys(
    table: pd.DataFrame,
    keys: Sequence[str],
    n: int = 10,
    ascending: bool | Sequence[bool] = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-key ranking: sort by ``keys[0]``, ties broken by ``keys[1]``, then
    ``keys[2]`` ... (ascending by default) and keep the first ``n`` rows.

    Rows with a missing value in any key cannot be ranked and are returned
    separately. Returns ``(selected, unranked)``; ``selected`` has a 1-based
    ``rank`` column.
    """
    keys = list(keys)
    missing_cols = [k for k in keys if k not in table.columns]
    if missing_cols:
        raise KeyError(
            f"Ranking columns not in table: {missing_cols}. "
            f"Available: {list(table.columns)}"
        )
    has_all = table[keys].notna().all(axis=1)
    ranked = (
        table[has_all]
        .sort_values(keys, ascending=ascending, kind="mergesort")
        .reset_index(drop=True)
    )
    selected = ranked.head(n).copy()
    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected, table[~has_all].reset_index(drop=True)


def select_top(
    table: pd.DataFrame,
    by: str = "haddock_score",
    n: int = 5,
    ascending: bool = True,
) -> pd.DataFrame:
    """Top ``n`` rows by a score column (HADDOCK score: lower is better)."""
    if by not in table.columns:
        raise KeyError(f"Column {by!r} not in table: {list(table.columns)}")
    return (
        table.dropna(subset=[by])
        .sort_values(by, ascending=ascending)
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trajectory(
    dcd_path: str | Path,
    topology_pdb: str | Path,
    stride: int = 1,
    heavy_only: bool = True,
):
    """Load a DCD with its topology PDB; hydrogens are dropped by default."""
    md = _mdtraj()
    dcd_path, topology_pdb = Path(dcd_path), Path(topology_pdb)
    for p in (dcd_path, topology_pdb):
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")

    traj = md.load_dcd(str(dcd_path), top=str(topology_pdb), stride=stride)
    if heavy_only:
        traj = traj.atom_slice(traj.topology.select("not element H"))
    return traj


def frame_labels(
    n_frames: int,
    minimized: bool = True,
    report_interval: Optional[int] = None,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Label the loaded frames: ``frame`` is the index in the original DCD
    (``loaded index * stride``), ``kind`` is 'initial'/'minimized'/'md' and
    ``time_ps`` the MD time (1 fs steps).
    """
    rows = []
    first_md = 2 if minimized else 1
    for j in range(n_frames):
        i = j * stride                    # index in the original DCD
        if i == 0:
            label, t = "initial", 0.0
        elif minimized and i == 1:
            label, t = "minimized", 0.0
        else:
            k = i - first_md + 1
            label = "md"
            t = k * report_interval * 0.001 if report_interval else np.nan
        rows.append({"frame": i, "kind": label, "time_ps": t})
    return pd.DataFrame(rows)


def chain_ids(traj) -> list[str]:
    return [
        str(getattr(c, "chain_id", None) or chr(ord("A") + c.index))
        for c in traj.topology.chains
    ]


def _chain_atoms(traj, chain_index: int) -> np.ndarray:
    return np.array(
        [a.index for a in traj.topology.chain(chain_index).atoms], dtype=int
    )


def _ca(traj, atom_idx: np.ndarray) -> np.ndarray:
    return np.array(
        [i for i in atom_idx if traj.topology.atom(i).name == "CA"], dtype=int
    )


# ---------------------------------------------------------------------------
# Interface (distance based, shared by analysis and viewer)
# ---------------------------------------------------------------------------

def _interface_sets(xyz_nm, idx_a, idx_b, res_of_atom, cutoff_nm):
    from scipy.spatial import cKDTree

    ta, tb = cKDTree(xyz_nm[idx_a]), cKDTree(xyz_nm[idx_b])
    near = ta.query_ball_tree(tb, cutoff_nm)            # per A atom: B atoms
    a_res, b_res, pairs = set(), set(), 0
    for i, hits in enumerate(near):
        if hits:
            pairs += len(hits)
            a_res.add(res_of_atom[idx_a[i]])
            b_res.update(res_of_atom[idx_b[j]] for j in hits)
    return a_res, b_res, pairs


def interface_residue_ids(
    traj, frame: int = -1, cutoff_angstrom: float = 5.0
) -> dict[str, list[int]]:
    """
    Residues (``resSeq`` numbers) with a heavy atom within the cutoff of the
    other chain, for one frame. Returns ``{chain_id: [resSeq, ...]}``.
    """
    ids = chain_ids(traj)
    if len(ids) < 2:
        raise ValueError("Interface analysis needs at least two chains.")
    idx_a, idx_b = _chain_atoms(traj, 0), _chain_atoms(traj, 1)
    res_of_atom = np.array([a.residue.index for a in traj.topology.atoms])
    a_res, b_res, _ = _interface_sets(
        traj.xyz[frame], idx_a, idx_b, res_of_atom, cutoff_angstrom / 10.0
    )
    seq = {r.index: r.resSeq for r in traj.topology.residues}
    return {
        ids[0]: sorted(seq[r] for r in a_res),
        ids[1]: sorted(seq[r] for r in b_res),
    }


def interface_residues_mda(
    topology_pdb: str | Path,
    dcd_path: str | Path | None = None,
    frame: int = -1,
    cutoff_angstrom: float = 5.0,
) -> dict[str, list[int]]:
    """
    Same quantity via MDAnalysis' ``around`` selection (independent
    implementation, used as a cross-check of :func:`interface_residue_ids`).
    """
    mda = _mda()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        u = mda.Universe(str(topology_pdb), str(dcd_path)) if dcd_path \
            else mda.Universe(str(topology_pdb))
        u.trajectory[frame]
    chains = list(dict.fromkeys(u.atoms.chainIDs))
    if len(chains) < 2:
        raise ValueError("Interface analysis needs at least two chains.")
    a, b = chains[0], chains[1]

    def near(c, other):
        ag = u.select_atoms(
            f"(chainID {c} and not name H*) and around {cutoff_angstrom} "
            f"(chainID {other} and not name H*)",
            periodic=False, # ignore the dummy 1 A unit cell stored in the DCD/PDB
        )
        return sorted(set(int(r) for r in ag.resids))

    return {a: near(a, b), b: near(b, a)}


# ---------------------------------------------------------------------------
# Per-frame analysis (mdtraj)
# ---------------------------------------------------------------------------

def analyze_trajectory(
    traj,
    minimized: bool = True,
    report_interval: Optional[int] = None,
    contact_cutoff_angstrom: float = 4.5,
    compute_bsa: bool = True,
    sasa_points: int = 240,
    stride: int = 1,
) -> pd.DataFrame:
    """
    One row per loaded frame. Pass the ``stride`` used in
    :func:`load_trajectory` so that ``frame``/``kind``/``time_ps`` refer to
    the original DCD frames.

    rmsd_ca_nm / rmsd_ca_A_nm / rmsd_ca_B_nm
        CA RMSD to frame 0 (complex superposed on all CA; each chain
        superposed on its own CA).
    rg_nm              radius of gyration of the complex (heavy atoms).
    com_distance_nm    distance between the two chain centres of mass.
    min_distance_nm    closest inter-chain heavy-atom distance.
    atom_contacts      inter-chain heavy-atom pairs within the cutoff.
    interface_res_A/B  residues with an inter-chain contact within the cutoff.
    bsa_A2             buried surface area = SASA(A) + SASA(B) - SASA(AB)
                       (Shrake-Rupley, 1.4 A probe), in A^2.
    """
    md = _mdtraj()
    from scipy.spatial import cKDTree

    ids = chain_ids(traj)
    if len(ids) < 2:
        raise ValueError("Complex metrics need at least two chains.")

    idx_a, idx_b = _chain_atoms(traj, 0), _chain_atoms(traj, 1)
    ca_all = _ca(traj, np.arange(traj.n_atoms))
    ca_a, ca_b = _ca(traj, idx_a), _ca(traj, idx_b)
    res_of_atom = np.array([a.residue.index for a in traj.topology.atoms])
    cutoff_nm = contact_cutoff_angstrom / 10.0

    out = frame_labels(traj.n_frames, minimized, report_interval, stride)
    out["chain_A"], out["chain_B"] = ids[0], ids[1]

    def rmsd(indices):
        if len(indices) < 3:
            return np.full(traj.n_frames, np.nan)
        return md.rmsd(traj, traj, 0, atom_indices=indices)

    out["rmsd_ca_nm"] = rmsd(ca_all)
    out["rmsd_ca_A_nm"] = rmsd(ca_a)
    out["rmsd_ca_B_nm"] = rmsd(ca_b)
    out["rg_nm"] = md.compute_rg(traj)

    masses = np.array([a.element.mass for a in traj.topology.atoms])
    com = lambda xyz, idx: np.average(xyz[idx], axis=0, weights=masses[idx])

    com_d, min_d, contacts, ia, ib = [], [], [], [], []
    for f in range(traj.n_frames):
        xyz = traj.xyz[f]
        com_d.append(float(np.linalg.norm(com(xyz, idx_a) - com(xyz, idx_b))))
        d, _ = cKDTree(xyz[idx_b]).query(xyz[idx_a], k=1)
        min_d.append(float(d.min()))
        a_res, b_res, pairs = _interface_sets(
            xyz, idx_a, idx_b, res_of_atom, cutoff_nm
        )
        contacts.append(pairs)
        ia.append(len(a_res))
        ib.append(len(b_res))
    out["com_distance_nm"] = com_d
    out["min_distance_nm"] = min_d
    out["atom_contacts"] = contacts
    out["interface_res_A"] = ia
    out["interface_res_B"] = ib

    if compute_bsa:
        sasa = lambda t: md.shrake_rupley(
            t, mode="atom", n_sphere_points=sasa_points
        ).sum(axis=1)
        s_ab = sasa(traj)
        s_a = sasa(traj.atom_slice(idx_a))
        s_b = sasa(traj.atom_slice(idx_b))
        out["bsa_A2"] = (s_a + s_b - s_ab) * 100.0   # nm^2 -> A^2
    return out


def analyze_many(
    table: pd.DataFrame,
    minimized: bool = True,
    report_interval: Optional[int] = None,
    stride: int = 1,
    **analysis_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze every row of a ``discover_from_results`` /
    ``discover_trajectories`` table.

    Per-row ``minimized`` / ``report_interval`` columns (from the run settings)
    take precedence over the function arguments. Rows with ``usable == False``
    are skipped and reported in the summary.

    Returns (per-frame table for all trajectories, one-row-per-trajectory summary).
    Trajectories that cannot be loaded are listed in ``summary['error']``.
    """
    frames, summaries = [], []
    for row in table.to_dict("records"):
        pid = row["prediction_id"]
        base = {"prediction_id": pid, "fold_id": row.get("fold_id", pid[:8])}
        for k in ("haddock_score", "em_bsa", "minimized_potential_kj_mol"):
            if k in row:
                base[k] = row[k]
        if row.get("usable") is False:
            summaries.append({**base, "error": f"not usable: {row.get('reason', '')}"})
            continue

        row_min = row.get("minimized")
        row_min = minimized if row_min is None or pd.isna(row_min) else bool(row_min)
        row_int = row.get("report_interval")
        row_int = report_interval if row_int is None or pd.isna(row_int) else int(row_int)
        try:
            traj = load_trajectory(row["dcd_path"], row["topology_pdb"], stride)
            df = analyze_trajectory(
                traj, row_min, row_int, stride=stride, **analysis_kwargs
            )
        except Exception as exc:
            summaries.append({**base, "error": f"{type(exc).__name__}: {exc}"})
            continue

        df.insert(0, "prediction_id", pid)
        frames.append(df)
        first, last = df.iloc[0], df.iloc[-1]
        summ = {
            **base,
            "n_frames": len(df),
            "rmsd_ca_final_nm": last["rmsd_ca_nm"],
            "rg_final_nm": last["rg_nm"],
            "min_distance_final_nm": last["min_distance_nm"],
            "interface_res_final": last["interface_res_A"] + last["interface_res_B"],
            "atom_contacts_final": last["atom_contacts"],
            "error": "",
        }
        if "bsa_A2" in df:
            summ.update(bsa_initial_A2=first["bsa_A2"], bsa_final_A2=last["bsa_A2"])
        summaries.append(summ)

    all_frames = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_frames, pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Per-residue RMSF (MDAnalysis)
# ---------------------------------------------------------------------------

def rmsf_per_residue(
    topology_pdb: str | Path,
    dcd_path: str | Path,
    start: int = 0,
    stop: Optional[int] = None,
) -> pd.DataFrame:
    """
    CA RMSF in Angstrom after superposing every frame on frame 0 (CA).

    Use ``start=2`` to exclude the initial/minimized frames and look at the MD
    part only. RMSF needs at least two frames; with fewer than ~10 it is only
    indicative.
    """
    mda = _mda()
    from MDAnalysis.analysis import align, rms

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # DCDReader notice
        u = mda.Universe(str(topology_pdb), str(dcd_path))
        n = len(u.trajectory)
        stop = n if stop is None else stop
        if stop - start < 2:
            raise ValueError(
                f"RMSF needs >= 2 frames in [start={start}, stop={stop}) "
                f"but the trajectory has {n} frame(s); lower `start`."
            )

        ca = u.select_atoms("name CA")
        align.AlignTraj(u, u, select="name CA", in_memory=True).run()
        r = rms.RMSF(ca).run(start=start, stop=stop)

    return pd.DataFrame({
        "chain": [a.chainID for a in ca],
        "resid": [int(a.resid) for a in ca],
        "resname": [a.resname for a in ca],
        "rmsf_A": r.results.rmsf,
    })


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_analysis(df: pd.DataFrame, title: str = ""):
    """Six-panel overview of :func:`analyze_trajectory` output."""
    import matplotlib.pyplot as plt

    x = df["frame"]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharex=True)

    ax[0, 0].plot(x, df["rmsd_ca_nm"] * 10, "o-", label="complex")
    ax[0, 0].plot(x, df["rmsd_ca_A_nm"] * 10, "s--", label=f"chain {df['chain_A'].iloc[0]}")
    ax[0, 0].plot(x, df["rmsd_ca_B_nm"] * 10, "^--", label=f"chain {df['chain_B'].iloc[0]}")
    ax[0, 0].set(title="CA RMSD to frame 0", ylabel="Å"); ax[0, 0].legend()

    ax[0, 1].plot(x, df["rg_nm"] * 10, "o-", color="tab:green")
    ax[0, 1].set(title="Radius of gyration", ylabel="Å")

    ax[0, 2].plot(x, df["com_distance_nm"] * 10, "o-", label="COM–COM")
    ax[0, 2].plot(x, df["min_distance_nm"] * 10, "s--", label="closest atoms")
    ax[0, 2].set(title="Inter-chain distance", ylabel="Å"); ax[0, 2].legend()

    ax[1, 0].plot(x, df["interface_res_A"], "o-", label="chain A")
    ax[1, 0].plot(x, df["interface_res_B"], "s--", label="chain B")
    ax[1, 0].set(title="Interface residues", ylabel="count", xlabel="frame"); ax[1, 0].legend()

    ax[1, 1].plot(x, df["atom_contacts"], "o-", color="tab:purple")
    ax[1, 1].set(title="Inter-chain atom contacts", xlabel="frame")

    if "bsa_A2" in df:
        ax[1, 2].plot(x, df["bsa_A2"], "o-", color="tab:red")
    ax[1, 2].set(title="Buried surface area", ylabel="Å²", xlabel="frame")

    for a in ax.ravel():
        a.grid(alpha=0.3)
        for x_min in df.loc[df["kind"] == "minimized", "frame"]:
            a.axvline(x_min, color="gray", ls=":", lw=1)    # minimized frame
    fig.suptitle(title or df.get("prediction_id", pd.Series([""])).iloc[0])
    fig.tight_layout()
    return fig


def plot_rmsf(rmsf: pd.DataFrame, title: str = ""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for i, (chain, g) in enumerate(rmsf.groupby("chain", sort=False)):
        ax.plot(g["resid"], g["rmsf_A"], "o-", ms=3,
                color=CHAIN_COLORS[i % len(CHAIN_COLORS)], label=f"chain {chain}")
    ax.set(xlabel="residue", ylabel="CA RMSF (Å)", title=title or "Per-residue RMSF")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def to_multimodel_pdb(
    traj,
    frames: Optional[Sequence[int]] = None,
    align: bool = True,
    max_frames: int = 200,
) -> str:
    """
    Multi-model PDB text (``MODEL``/``ENDMDL``). Frames are superposed on the
    CA atoms of frame 0 so the animation does not drift.
    """
    if frames is None:
        step = max(1, int(np.ceil(traj.n_frames / max_frames)))
        frames = list(range(0, traj.n_frames, step))
    sub = traj[list(frames)]
    if align and sub.n_frames > 1:
        sub = sub.superpose(sub, 0, atom_indices=_ca(sub, np.arange(sub.n_atoms)))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "view.pdb"
        sub.save_pdb(str(path))
        return path.read_text()


def export_multimodel_pdb(traj, path: str | Path, **kwargs) -> Path:
    """Write a multi-model PDB (opens in PyMOL / ChimeraX / VMD)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_multimodel_pdb(traj, **kwargs))
    return path


def save_analysis(df: pd.DataFrame, out_dir: str | Path, name: str, fig=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out_dir / f"{name}_analysis.csv"}
    df.to_csv(paths["csv"], index=False)
    if fig is not None:
        paths["png"] = out_dir / f"{name}_analysis.png"
        fig.savefig(paths["png"], dpi=150)
    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------
# py3Dmol views
# ---------------------------------------------------------------------------

def _style_view(view, ids, interface, style, viewer=None, show_interface=True):
    kw = {"viewer": viewer} if viewer is not None else {}
    for i, cid in enumerate(ids):
        color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        spec = {style: {"color": color}} if style != "stick" \
            else {"stick": {"color": color, "radius": 0.15}}
        view.setStyle({"chain": cid}, spec, **kw)
        if show_interface and interface.get(cid):
            view.addStyle(
                {"chain": cid, "resi": interface[cid]},
                {"stick": {"radius": 0.14, "colorscheme": "default"}},
                **kw,
            )


def show_trajectory(
    traj,
    style: str = "cartoon",
    show_interface: bool = True,
    interface_cutoff_angstrom: float = 5.0,
    interface_frame: int = -1,
    interval_ms: int = 200,
    animate: bool = True,
    max_frames: int = 200,
    width: int = 800,
    height: int = 600,
):
    """
    Animated py3Dmol view of all frames (chains coloured, interface residues as
    sticks). Call ``.show()`` on the returned view (Colab displays it
    automatically when it is the last expression of a cell).
    """
    py3Dmol = _py3dmol()
    ids = chain_ids(traj)
    interface = (
        interface_residue_ids(traj, interface_frame, interface_cutoff_angstrom)
        if show_interface and len(ids) >= 2 else {}
    )
    pdb = to_multimodel_pdb(traj, max_frames=max_frames)

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(pdb, "pdb")
    _style_view(view, ids, interface, style, show_interface=show_interface)
    view.zoomTo()
    if animate and traj.n_frames > 1:
        view.animate({"interval": interval_ms, "loop": "forward"})
    return view


def show_frame(traj, frame: int = -1, **kwargs):
    """Static view of a single frame."""
    return show_trajectory(traj[frame], animate=False, interface_frame=0, **kwargs)


def show_grid(
    trajs: Sequence[Any],
    titles: Sequence[str] | None = None,
    frame: int = -1,
    columns: int = 2,
    style: str = "cartoon",
    interface_cutoff_angstrom: float = 5.0,
    cell_width: int = 420,
    cell_height: int = 380,
):
    """Side-by-side static comparison of one frame from several trajectories."""
    py3Dmol = _py3dmol()
    n = len(trajs)
    rows = int(np.ceil(n / columns))
    view = py3Dmol.view(
        width=cell_width * columns, height=cell_height * rows,
        viewergrid=(rows, columns), linked=False,
    )
    for k, traj in enumerate(trajs):
        pos = (k // columns, k % columns)
        one = traj[frame]
        ids = chain_ids(one)
        interface = interface_residue_ids(one, 0, interface_cutoff_angstrom) \
            if len(ids) >= 2 else {}
        view.addModel(to_multimodel_pdb(one, align=False), "pdb", viewer=pos)
        _style_view(view, ids, interface, style, viewer=pos)
        if titles:
            view.addLabel(
                titles[k],
                {"fontSize": 12, "backgroundOpacity": 0.4,
                 "position": {"x": 0, "y": 0, "z": 0}, "useScreen": True,
                 "screenOffset": {"x": 6, "y": 6}},
                viewer=pos,
            )
        view.zoomTo(viewer=pos)
    return view


if __name__ == "__main__":
    print(f"trajectory_viewer.py {MODULE_VERSION}")
