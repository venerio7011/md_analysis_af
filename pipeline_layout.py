# -*- coding: utf-8 -*-
"""
pipeline_layout.py  (v1.4.0)

One place that defines where every file of the AF3 -> HADDOCK -> OpenMM
pipeline lives, shared by both driver notebooks
(AF3_MD_pipeline_kneedle.ipynb and DCD_trajectory_viewer_kneedle.ipynb).

Layouts
-------
project (default) - one folder per sample set:

    <DRIVE>/AF3_HADDOCK/
    ├── modules/                       python modules (shared)
    ├── charmm/                        CHARMM rtf/prm (shared)
    └── projects/<DATASET>/
        ├── inbox/                     batch zips to import (upload / Drive)
        ├── input/                     AF3 server results: fold_001/ ... fold_NNN/
        ├── work/                      intermediates  -> removed by cleanup_work()
        │   ├── 1_structure_pdb/
        │   ├── 2_air/
        │   ├── 3_haddock_ready/
        │   ├── 4_emscoring/
        │   └── 5_openmm/              results/ trajectories/ pdb/ (+ *_md, psf ...)
        ├── outputs/                   kept
        │   ├── pipeline/              merged scores, evidence tables, samples
        │   ├── viewer/                selection, per-frame analysis, plots, PDBs
        │   ├── trajectories/          DCD + topology PDB of the selected samples
        │   └── cleanup_log.json
        └── pipeline_status.json       stage markers (pipeline, viewer, cleanup)

legacy - the original folders at the Drive root (MD_output, MD_output_pdb,
MD_output_air, MD_output_haddock, MD_output_emscoring, MD_output_openmm);
outputs/ and pipeline_status.json are placed in projects/<DATASET>/.

migrate_legacy() moves an existing legacy run into the project layout and
rewrites the absolute paths stored in its manifests/CSVs, so finished steps
are still skipped afterwards.

import_af3_batch() (v1.1.0) imports AF3 Server results delivered as a batch
(zip or folder: <batch>/<NNN_jobname>/fold_NNN_<jobname>_*.{cif,json}) into
input/fold_NNN/, taking the fold id from the AF3 file names and copying only
the files the pipeline reads (msas/, templates/, __MACOSX are skipped).
detect_folds() lists the fold_NNN folders of an input directory.
v1.2.0: ``offset`` renumbers a batch (e.g. a second batch that starts again
at 001 -> fold_031 ...; ``offset="auto"`` continues after the highest
existing number). Jobs already imported (same AF3 job name) are skipped, so
re-running an import never duplicates samples; fold_map.csv records
fold_id <- original number / job / source. projects/<DATASET>/inbox/ is a
drop folder: find_batch_sources() lists the batch zips/folders found there
and store_upload() puts a file uploaded in Colab into it.

v1.3.0: resolve_project() picks the project folder the user created
(auto-detect when only one exists, exact name or full path otherwise) and
never creates a new one unless asked; check_layout() warns about
sub-folders whose names are close to, but not exactly, input / inbox /
work / outputs.

v1.4.0: import_from_input() converts AF3 material uploaded straight into
input/ (batch zips, extracted batch folders or sample folders) into
input/fold_NNN/ and moves the originals to input/_raw_imported/.

cleanup_work() removes ONLY the explicit intermediate folders of a layout
(never input/, modules/, charmm/ or outputs/) and only after both notebooks
have marked their stage as done and the final outputs exist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

MODULE_VERSION = "1.4.0"

LAYOUT_MODES = ("project", "legacy")

# legacy folder name -> attribute of Layout
_LEGACY_NAMES = {
    "input_dir": "MD_output",
    "structure_dir": "MD_output_pdb",
    "air_dir": "MD_output_air",
    "haddock_dir": "MD_output_haddock",
    "emscoring_dir": "MD_output_emscoring",
    "openmm_dir": "MD_output_openmm",
}
_PROJECT_WORK_NAMES = {
    "structure_dir": "1_structure_pdb",
    "air_dir": "2_air",
    "haddock_dir": "3_haddock_ready",
    "emscoring_dir": "4_emscoring",
    "openmm_dir": "5_openmm",
}
_WORK_ATTRS = tuple(_PROJECT_WORK_NAMES)

# Final files that must exist before cleanup (relative to outputs_dir).
REQUIRED_OUTPUTS = {
    "pipeline": ("pipeline/em_openmm_merged.csv",),
    "viewer": ("viewer/selection_kneedle_and.csv", "trajectories/selected_trajectories.csv"),
}

_TEXT_SUFFIXES = {".json", ".csv", ".cfg", ".toml"}
_MAX_REWRITE_BYTES = 50 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    mode: str
    dataset: str
    drive: Path
    module_dir: Path
    charmm_dir: Path
    project_dir: Path
    input_dir: Path
    work_dir: Optional[Path]          # None in legacy mode (folders at Drive root)
    structure_dir: Path
    air_dir: Path
    haddock_dir: Path
    emscoring_dir: Path
    openmm_dir: Path
    outputs_dir: Path
    status_file: Path
    inbox_dir: Path

    @property
    def work_dirs(self) -> list[Path]:
        """Intermediate folders; the only ones cleanup_work() may delete."""
        return [getattr(self, a) for a in _WORK_ATTRS]

    @property
    def outputs_pipeline(self) -> Path:
        return self.outputs_dir / "pipeline"

    @property
    def outputs_viewer(self) -> Path:
        return self.outputs_dir / "viewer"

    @property
    def outputs_trajectories(self) -> Path:
        return self.outputs_dir / "trajectories"

    def ensure(self) -> "Layout":
        """Create the output folders (the input folder is never created here)."""
        for p in [*self.work_dirs, self.outputs_pipeline, self.outputs_viewer,
                  self.outputs_trajectories]:
            p.mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> str:
        rows = [
            ("mode", self.mode), ("dataset", self.dataset),
            ("modules", self.module_dir), ("inbox", self.inbox_dir),
            ("input (AF3)", self.input_dir),
            ("structure", self.structure_dir), ("air", self.air_dir),
            ("haddock", self.haddock_dir), ("emscoring", self.emscoring_dir),
            ("openmm", self.openmm_dir), ("outputs", self.outputs_dir),
        ]
        return "\n".join(f"{k:12s}: {v}" for k, v in rows)


def make_layout(
    drive: str | Path,
    dataset: str,
    mode: str = "project",
    input_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> Layout:
    """
    Paths of one dataset. ``input_dir`` overrides the AF3 input folder
    (default: projects/<dataset>/input, or MyDrive/MD_output in legacy mode).
    ``project_dir`` uses an existing folder anywhere on Drive instead of
    AF3_HADDOCK/projects/<dataset> (see resolve_project()).
    """
    if project_dir is not None:
        dataset = Path(project_dir).name
    if mode not in LAYOUT_MODES:
        raise ValueError(f"mode must be one of {LAYOUT_MODES}, got {mode!r}")
    if not dataset or re.search(r"[\\/]", dataset):
        raise ValueError(f"dataset must be a plain folder name, got {dataset!r}")

    drive = Path(drive)
    base = drive / "AF3_HADDOCK"
    project = Path(project_dir) if project_dir is not None else base / "projects" / dataset

    if mode == "project":
        work = project / "work"
        dirs = {a: work / n for a, n in _PROJECT_WORK_NAMES.items()}
        default_input = project / "input"
    else:
        work = None
        dirs = {a: drive / _LEGACY_NAMES[a] for a in _WORK_ATTRS}
        default_input = drive / _LEGACY_NAMES["input_dir"]

    return Layout(
        mode=mode,
        dataset=dataset,
        drive=drive,
        module_dir=base / "modules",
        charmm_dir=base / "charmm",
        project_dir=project,
        input_dir=Path(input_dir) if input_dir else default_input,
        work_dir=work,
        outputs_dir=project / "outputs",
        status_file=project / "pipeline_status.json",
        inbox_dir=project / "inbox",
        **dirs,
    )


def _norm_name(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name).casefold()


def list_projects(drive: str | Path) -> list[str]:
    """Folder names in AF3_HADDOCK/projects/ (hidden ones excluded)."""
    base = Path(drive) / "AF3_HADDOCK" / "projects"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))


def resolve_project(
    drive: str | Path,
    dataset: Optional[str] = None,
    project_dir: str | Path | None = None,
    create: bool = False,
) -> Path:
    """
    The project folder to use; never creates one unless ``create=True``.

    project_dir   full path of a folder you created (anywhere on Drive)
    dataset       folder name in AF3_HADDOCK/projects/ (exact name)
    neither       the only folder in AF3_HADDOCK/projects/
    A missing folder raises with the existing names and close matches
    (case / spaces / '_' '-' differ), so a typo never silently creates a
    second project.
    """
    drive = Path(drive)
    base = drive / "AF3_HADDOCK" / "projects"
    existing = list_projects(drive)

    if project_dir is not None:
        p = Path(project_dir)
        if p.is_dir():
            return p
        if create:
            p.mkdir(parents=True)
            return p
        raise FileNotFoundError(
            f"PROJECT_DIR does not exist: {p}\n"
            "Check the path in Google Drive (Colab sees it under /content/drive/MyDrive/...), "
            "or set CREATE_NEW_DATASET = True to create it."
        )

    if not dataset:
        if len(existing) == 1:
            return base / existing[0]
        if not existing:
            raise FileNotFoundError(
                f"No project folder in {base}.\n"
                "Create one in Google Drive (e.g. AF3_HADDOCK/projects/ST307) or set "
                "DATASET = 'ST307' and CREATE_NEW_DATASET = True."
            )
        raise RuntimeError(
            f"Several project folders in {base}: {existing}\n"
            "Set DATASET to one of them."
        )

    if re.search(r"[\\/]", dataset):
        raise ValueError(f"DATASET must be a folder name, not a path: {dataset!r} (use PROJECT_DIR)")
    p = base / dataset
    if p.is_dir():
        return p
    close = [n for n in existing if _norm_name(n) == _norm_name(dataset)]
    if create and not close:
        p.mkdir(parents=True)
        return p
    hint = (f"Did you mean {close}? (names differ in case, spaces or '_'/'-')" if close
            else f"Existing project folders: {existing or 'none'}")
    raise FileNotFoundError(
        f"Project folder not found: {p}\n{hint}\n"
        "Set DATASET to the exact folder name"
        + ("" if close else ", or CREATE_NEW_DATASET = True to create it") + "."
    )


_EXPECTED_SUBDIRS = ("inbox", "input", "work", "outputs")


def check_layout(layout: Layout) -> list[str]:
    """
    Warnings about the project folder: sub-folders whose names are close to,
    but not exactly, the expected ones (e.g. 'output', 'Input', 'outputs ')
    - the pipeline would create and use a separate folder with the exact name.
    """
    if layout.mode != "project" or not layout.project_dir.is_dir():
        return []
    warnings_ = []
    present = [p.name for p in layout.project_dir.iterdir() if p.is_dir()]
    for want in _EXPECTED_SUBDIRS:
        if want in present:
            continue
        near = [n for n in present
                if _norm_name(n) in (_norm_name(want), _norm_name(want.rstrip("s")),
                                     _norm_name(want + "s"))]
        if near:
            warnings_.append(
                f"'{want}' is expected but the project has {near}; "
                f"rename it to '{want}' in Google Drive, otherwise a separate '{want}' folder is used."
            )
    if layout.input_dir.is_dir():
        raw = {p.name for p in find_batch_sources(layout.input_dir)}
        odd = [p.name for p in layout.input_dir.iterdir()
               if p.is_dir() and not p.name.startswith((".", "_"))
               and not re.fullmatch(r"fold_\d{3}", p.name) and p.name not in raw]
        if odd:
            warnings_.append(
                f"input/ contains folders that are not fold_NNN and hold no AF3 results "
                f"(ignored): {odd[:5]}" + (" ..." if len(odd) > 5 else "")
            )
    return warnings_


def legacy_data_present(drive: str | Path) -> list[str]:
    """Names of legacy MD_output* folders that exist at the Drive root."""
    drive = Path(drive)
    return [n for n in _LEGACY_NAMES.values() if (drive / n).is_dir()]


def detect_folds(input_dir: str | Path) -> tuple[list[str], list[str]]:
    """
    (fold ids, other names): ``fold_NNN`` folders (three digits) in the input
    directory, sorted, and folder names that look like folds but do not match
    the three-digit pattern (e.g. fold_1000) and would be ignored.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return [], []
    folds, other = [], []
    for p in input_dir.iterdir():
        if p.is_dir() and p.name.startswith("fold_"):
            (folds if re.fullmatch(r"fold_\d{3}", p.name) else other).append(p.name)
    return sorted(folds), sorted(other)


# ---------------------------------------------------------------------------
# Import of AF3 Server batches
# ---------------------------------------------------------------------------

_AF3_FILE = re.compile(
    r"^(fold_\d+)_.+?_(?:model_(\d+)\.cif|summary_confidences_(\d+)\.json"
    r"|full_data_(\d+)\.json|job_request\.json)$"
)
_SKIP_PARTS = ("__MACOSX", "msas", "templates")


def _batch_entries(source: Path):
    """Yield (sample folder, file name, size, opener) for every file of a batch."""
    if source.is_file() and source.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(source)
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2 or any(x in _SKIP_PARTS for x in parts) or parts[-1].startswith("._"):
                continue
            yield parts[-2], parts[-1], info.file_size, (lambda i=info: zf.open(i))
    elif source.is_dir():
        for root, dirs, files in os.walk(source):
            dirs[:] = [d for d in dirs if d not in _SKIP_PARTS and not d.startswith("_")
                       and not re.fullmatch(r"fold_\d{3}", d)]
            for name in files:
                if name.startswith("._"):
                    continue
                f = Path(root) / name
                yield f.parent.name, name, f.stat().st_size, (lambda f=f: open(f, "rb"))
    else:
        raise FileNotFoundError(f"Batch source not found (zip or folder): {source}")


def _job_key(file_name: str) -> Optional[str]:
    """AF3 job name of a *_full_data_0.json file: fold_001_<job>_full_data_0.json -> <job>."""
    m = re.match(r"^fold_\d+_(.+)_full_data_0\.json$", file_name)
    return m.group(1) if m else None


def _existing_jobs(input_dir: Path) -> dict[str, str]:
    """job name -> fold id for every sample already in input_dir."""
    jobs = {}
    for fold in detect_folds(input_dir)[0]:
        for f in (input_dir / fold).glob("*_full_data_0.json"):
            key = _job_key(f.name)
            if key:
                jobs[key] = fold
    return jobs


def _fold_num(fold_id: str) -> int:
    return int(fold_id.split("_")[1])


def import_af3_batch(
    source: str | Path,
    input_dir: str | Path,
    dry_run: bool = True,
    all_full_data: bool = False,
    offset: int | str = 0,
    planned: Iterable[str] = (),
    label: Optional[str] = None,
) -> dict[str, Any]:
    """
    Import an AF3 Server batch into ``input_dir/fold_NNN/``.

    The fold id comes from the AF3 file names (``fold_NNN_<job>_...``), not
    from the folder name. Copied per sample: ``*_model_N.cif``,
    ``*_summary_confidences_N.json``, ``*_job_request.json`` and
    ``*_full_data_0.json`` (the only full_data the pipeline reads; all five
    with ``all_full_data=True``).

    offset
        0       keep the AF3 numbers (fold_001 stays fold_001)
        N       renumber: fold_001 -> fold_{1+N:03d}; the files are renamed
                too (fold_001_<job>_... -> fold_031_<job>_...), because the
                pipeline derives the fold of a prediction from its file name
        "auto"  0 if the batch numbers are free, otherwise continue after the
                highest fold number already used (input + ``planned``)
    planned
        fold ids reserved by imports planned earlier in the same dry run, so
        several batches can be previewed together.

    Jobs whose AF3 job name already exists in ``input_dir`` are skipped
    (re-running an import never duplicates a sample). Problems (missing
    files, two jobs with the same number, a target number in use, numbers
    above 999) stop the import before anything is copied.
    """
    source, input_dir = Path(source), Path(input_dir)
    label = label or source.name
    folds: dict[str, dict[str, Any]] = {}
    for sample, name, size, opener in _batch_entries(source):
        m = _AF3_FILE.match(name)
        if not m:
            continue
        fold, model_i, summ_i, full_i = m.group(1), m.group(2), m.group(3), m.group(4)
        if full_i is not None and full_i != "0" and not all_full_data:
            continue
        rec = folds.setdefault(fold, {"samples": set(), "files": {}, "models": set(),
                                      "summaries": set(), "job": None})
        rec["samples"].add(sample)
        rec["files"][name] = (size, opener)
        if model_i is not None:
            rec["models"].add(model_i)
        if summ_i is not None:
            rec["summaries"].add(summ_i)
        if full_i == "0":
            rec["job"] = _job_key(name)

    problems: list[str] = []
    if not folds:
        problems.append("no AF3 result files (fold_NNN_*_full_data_0.json ...) found")

    existing_jobs = _existing_jobs(input_dir)
    used = {_fold_num(f) for f in detect_folds(input_dir)[0]} | {_fold_num(f) for f in planned}

    skipped, todo = [], []
    for fold in sorted(folds, key=_fold_num):
        rec = folds[fold]
        if rec["job"] and rec["job"] in existing_jobs:
            skipped.append({"original": fold, "fold_id": existing_jobs[rec["job"]], "job": rec["job"]})
        else:
            todo.append(fold)

    if offset == "auto":
        clash = any(_fold_num(f) in used for f in todo)
        off = max(used) if (clash and used) else 0
    else:
        off = int(offset)
        if off < 0:
            problems.append("offset must be >= 0")

    plan = []
    for fold in todo:
        rec = folds[fold]
        new_num = _fold_num(fold) + off
        new_fold = f"fold_{new_num:03d}"
        names = set(rec["files"])
        tag = f"{fold}->{new_fold}" if off else fold
        if not 1 <= new_num <= 999:
            problems.append(f"{tag}: fold number must be 001 ... 999")
        if len(rec["samples"]) > 1:
            problems.append(f"{tag}: several AF3 jobs use this number: {sorted(rec['samples'])}")
        if not rec["job"]:
            problems.append(f"{tag}: *_full_data_0.json missing")
        if not rec["models"]:
            problems.append(f"{tag}: no *_model_N.cif")
        missing_summ = sorted(rec["models"] - rec["summaries"])
        if missing_summ:
            problems.append(f"{tag}: summary_confidences missing for model(s) {missing_summ}")
        if new_num in used:
            holder = next((j for j, f in existing_jobs.items() if f == new_fold), None)
            used_by = f"job {holder}" if holder else "another batch in this import"
            problems.append(f"{tag}: {new_fold} is already used by {used_by} "
                            f"(use offset='auto' or a larger offset)")
        used.add(new_num)
        size = sum(v[0] for v in rec["files"].values())
        plan.append({"fold_id": new_fold, "original": fold, "job": rec["job"],
                     "sample": sorted(rec["samples"])[0], "files": len(names),
                     "bytes": size, "size": _fmt(size)})

    total = sum(p["bytes"] for p in plan)
    report = {"dry_run": dry_run, "source": str(source), "input_dir": str(input_dir),
              "offset": off,
              "folds": plan, "skipped_existing": skipped, "problems": problems,
              "total_bytes": total, "total_size": _fmt(total), "imported": 0}
    if dry_run or problems:
        return report

    for p in plan:
        target = input_dir / p["fold_id"]
        target.mkdir(parents=True, exist_ok=True)
        old_prefix, new_prefix = p["original"] + "_", p["fold_id"] + "_"
        for name, (_, opener) in folds[p["original"]]["files"].items():
            out_name = new_prefix + name[len(old_prefix):] if off else name
            with opener() as src, open(target / out_name, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        report["imported"] += 1

    import csv
    fold_map = input_dir / "fold_map.csv"
    new_file = not fold_map.is_file()
    with fold_map.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["fold_id", "original_fold", "job", "sample_folder", "source", "imported_utc"])
        for p in plan:
            w.writerow([p["fold_id"], p["original"], p["job"], p["sample"], label, _now()])

    log = input_dir / "import_log.json"
    history = json.loads(log.read_text()) if log.is_file() else []
    history.append({"utc": _now(), "source": label, "offset": off,
                    "imported": [f"{p['original']}->{p['fold_id']}" for p in plan],
                    "skipped_existing": [x["original"] for x in skipped], "size": _fmt(total)})
    log.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return report


def find_batch_sources(inbox_dir: str | Path) -> list[Path]:
    """
    Batch zips / extracted batch folders in ``inbox_dir`` that contain AF3
    results (``fold_NNN_*_full_data_0.json``), sorted by name. The
    ``imported/`` subfolder is ignored.
    """
    inbox_dir = Path(inbox_dir)
    if not inbox_dir.is_dir():
        return []
    found = []
    for p in sorted(inbox_dir.iterdir()):
        if p.name.startswith((".", "_")) or p.name == "imported" or re.fullmatch(r"fold_\d{3}", p.name):
            continue
        try:
            if p.is_file() and p.suffix.lower() == ".zip":
                with zipfile.ZipFile(p) as zf:
                    ok = any(n.endswith("_full_data_0.json") and "__MACOSX" not in n
                             for n in zf.namelist())
            elif p.is_dir():
                ok = next(p.rglob("fold_*_full_data_0.json"), None) is not None
            else:
                ok = False
        except zipfile.BadZipFile:
            ok = False
        if ok:
            found.append(p)
    return found


def store_upload(path: str | Path, inbox_dir: str | Path) -> Path:
    """Move a file (e.g. uploaded to /content in Colab) into the inbox; never overwrites."""
    path, inbox_dir = Path(path), Path(inbox_dir)
    inbox_dir.mkdir(parents=True, exist_ok=True)
    dst = inbox_dir / path.name
    k = 1
    while dst.exists():
        dst = inbox_dir / f"{path.stem}_{k}{path.suffix}"
        k += 1
    shutil.move(str(path), str(dst))
    return dst


RAW_ARCHIVE = "_raw_imported"


def raw_input_sources(input_dir: str | Path) -> dict[str, list[Path]]:
    """
    AF3 material uploaded straight into input/: {'zips': [...], 'folders': [...]}.
    fold_NNN folders, '_'/'.'-prefixed folders and files other than zips are not
    sources.
    """
    input_dir = Path(input_dir)
    found = find_batch_sources(input_dir) if input_dir.is_dir() else []
    return {"zips": [p for p in found if p.is_file()],
            "folders": [p for p in found if p.is_dir()]}


def import_from_input(
    input_dir: str | Path,
    offset: int | str = "auto",
    dry_run: bool = False,
    all_full_data: bool = False,
    raw_after_import: str = "archive",
) -> dict[str, Any]:
    """
    Convert AF3 results uploaded straight into ``input_dir`` into
    ``input_dir/fold_NNN/`` (same rules as import_af3_batch()).

    Every zip is one batch; all loose folders together (a batch folder or the
    sample folders themselves) form one batch, so a second batch uploaded as
    sample folders 001... is renumbered as a whole. Nothing is changed when
    any batch has a problem. Afterwards the originals are moved to
    ``input/_raw_imported/`` (``raw_after_import="archive"``), deleted
    (``"delete"``) or left in place (``"keep"``; they are skipped as already
    imported on the next run).
    """
    if raw_after_import not in ("archive", "delete", "keep"):
        raise ValueError("raw_after_import must be 'archive', 'delete' or 'keep'")
    input_dir = Path(input_dir)
    src = raw_input_sources(input_dir)
    jobs = [(z, z.name) for z in src["zips"]]
    if src["folders"]:
        jobs.append((input_dir, "input/: " + ", ".join(p.name for p in src["folders"])))

    # 1) plan everything first (dry run), stop on any problem
    reports, planned = [], []
    for path, label in jobs:
        r = import_af3_batch(path, input_dir, dry_run=True, all_full_data=all_full_data,
                             offset=offset, planned=planned, label=label)
        r["label"] = label
        reports.append(r)
        planned += [f["fold_id"] for f in r["folds"]]
    problems = [f"{r['label']}: {p}" for r in reports for p in r["problems"]]
    result = {"dry_run": dry_run, "batches": reports, "problems": problems,
              "imported": 0, "raw": [str(p) for p in src["zips"] + src["folders"]], "moved_raw": []}
    if dry_run or problems or not jobs:
        return result

    # 2) import (offset per batch was fixed in the plan)
    for (path, label), plan in zip(jobs, reports, strict=True):
        r = import_af3_batch(path, input_dir, dry_run=False, all_full_data=all_full_data,
                             offset=plan["offset"], label=label)
        if r["problems"]:
            raise RuntimeError(f"{label}: import failed after planning: {r['problems']}")
        result["imported"] += r["imported"]

    # 3) originals
    for raw in src["zips"] + src["folders"]:
        if raw_after_import == "archive":
            dst_dir = input_dir / RAW_ARCHIVE
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / raw.name
            if dst.exists():
                dst = dst_dir / f"{raw.stem}_{_now().replace(':', '')}{raw.suffix}"
            shutil.move(str(raw), str(dst))
            result["moved_raw"].append(str(dst))
        elif raw_after_import == "delete":
            shutil.rmtree(raw) if raw.is_dir() else raw.unlink()
            result["moved_raw"].append(f"deleted {raw.name}")
    return result


def archive_source(path: str | Path, inbox_dir: str | Path) -> Path:
    """Move an imported batch from the inbox to inbox/imported/."""
    path = Path(path)
    dst_dir = Path(inbox_dir) / "imported"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / path.name
    if dst.exists():
        dst = dst_dir / f"{path.stem}_{_now().replace(':', '')}{path.suffix}"
    shutil.move(str(path), str(dst))
    return dst


# ---------------------------------------------------------------------------
# Stage markers
# ---------------------------------------------------------------------------

def read_status(layout: Layout) -> dict[str, Any]:
    try:
        return json.loads(layout.status_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def mark_stage(layout: Layout, stage: str, **info: Any) -> dict[str, Any]:
    """Record that a stage ('pipeline', 'viewer', 'cleanup', ...) finished."""
    status = read_status(layout)
    status[stage] = {"done_utc": _now(), **info}
    layout.status_file.parent.mkdir(parents=True, exist_ok=True)
    layout.status_file.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return status


def stage_done(layout: Layout, stage: str) -> bool:
    return stage in read_status(layout)


# ---------------------------------------------------------------------------
# Collecting final outputs
# ---------------------------------------------------------------------------

def collect_files(files: Iterable[str | Path], target_dir: str | Path) -> list[Path]:
    """Copy existing files into ``target_dir``; missing ones are skipped."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        f = Path(f)
        if f.is_file():
            dst = target_dir / f.name
            shutil.copy2(f, dst)
            out.append(dst)
    return out


def collect_trajectories(
    layout: Layout,
    selected,
    id_col: str = "prediction_id",
) -> "Any":
    """
    Copy DCD + topology PDB of the selected samples into
    outputs/trajectories/ and write selected_trajectories.csv with the new
    paths. ``selected`` is a pandas DataFrame with dcd_path / topology_pdb.
    Returns the written table.
    """
    import pandas as pd

    dst_dir = layout.outputs_trajectories
    dst_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in selected.to_dict("records"):
        row = dict(r)
        for col in ("dcd_path", "topology_pdb"):
            src = Path(str(r.get(col) or ""))
            if src.is_file():
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                row[col] = str(dst)
                row[f"{col}_copied"] = True
            else:
                row[f"{col}_copied"] = False
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(dst_dir / "selected_trajectories.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------

def folder_size(path: Path) -> tuple[int, int]:
    """(bytes, file count) of a folder; (0, 0) if it does not exist."""
    total = n = 0
    if path.is_dir():
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                    n += 1
                except OSError:
                    pass
    return total, n


def _fmt(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}" if unit != "B" else f"{b} B"
        b /= 1024
    return f"{b:.1f} TB"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _check_deletable(layout: Layout, path: Path) -> None:
    protected = [layout.input_dir, layout.module_dir, layout.charmm_dir,
                 layout.outputs_dir, layout.status_file]
    for p in protected:
        if path.resolve() == p.resolve() or _is_within(p, path) or _is_within(path, p):
            raise RuntimeError(f"Refusing to delete {path}: overlaps protected folder {p}")
    if not _is_within(path, layout.drive):
        raise RuntimeError(f"Refusing to delete {path}: outside {layout.drive}")
    if path.resolve() == layout.drive.resolve():
        raise RuntimeError("Refusing to delete the Drive root.")


def cleanup_work(
    layout: Layout,
    dry_run: bool = True,
    require_stages: tuple[str, ...] = ("pipeline", "viewer"),
    force: bool = False,
) -> dict[str, Any]:
    """
    Remove the intermediate folders (``layout.work_dirs``).

    Without ``force`` it only proceeds when every stage in ``require_stages``
    is marked done and the files in REQUIRED_OUTPUTS exist. ``dry_run=True``
    (default) only reports what would be deleted. The input, modules, charmm
    and outputs folders are never touched.
    """
    problems = []
    if not force:
        status = read_status(layout)
        for st in require_stages:
            if st not in status:
                problems.append(f"stage '{st}' is not marked done")
            for rel in REQUIRED_OUTPUTS.get(st, ()):
                if not (layout.outputs_dir / rel).is_file():
                    problems.append(f"missing final output: outputs/{rel}")

    plan = []
    for d in layout.work_dirs:
        size, n = folder_size(d)
        plan.append({"path": str(d), "exists": d.is_dir(), "bytes": size, "files": n,
                     "size": _fmt(size)})
    total = sum(p["bytes"] for p in plan)
    report = {"dry_run": dry_run, "problems": problems, "plan": plan,
              "total_bytes": total, "total_size": _fmt(total), "deleted": []}

    if problems or dry_run:
        return report

    for d in layout.work_dirs:
        if d.is_dir():
            _check_deletable(layout, d)
    for d in layout.work_dirs:
        if d.is_dir():
            shutil.rmtree(d)
            report["deleted"].append(str(d))
    if layout.work_dir is not None and layout.work_dir.is_dir() and not any(layout.work_dir.iterdir()):
        layout.work_dir.rmdir()

    layout.outputs_dir.mkdir(parents=True, exist_ok=True)
    log = layout.outputs_dir / "cleanup_log.json"
    history = json.loads(log.read_text()) if log.is_file() else []
    history.append({"utc": _now(), "deleted": report["deleted"],
                    "freed_bytes": total, "freed": _fmt(total)})
    log.write_text(json.dumps(history, indent=2), encoding="utf-8")
    mark_stage(layout, "cleanup", freed=_fmt(total), deleted=len(report["deleted"]))
    return report


# ---------------------------------------------------------------------------
# Migration legacy -> project
# ---------------------------------------------------------------------------

def _rewrite_paths(root: Path, mapping: list[tuple[str, str]]) -> int:
    """Replace absolute legacy path prefixes in small text files under root."""
    if not root.is_dir():
        return 0
    patterns = [
        (re.compile(re.escape(old) + r"(?=[/\\\"',\s]|$)"), new)
        for old, new in sorted(mapping, key=lambda m: -len(m[0]))
    ]
    changed = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            if p.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            try:
                if p.stat().st_size > _MAX_REWRITE_BYTES:
                    continue
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            new_text = text
            for pat, new in patterns:
                new_text = pat.sub(new, new_text)
            if new_text != text:
                p.write_text(new_text, encoding="utf-8")
                changed += 1
    return changed


def migrate_legacy(
    drive: str | Path,
    dataset: str,
    dry_run: bool = True,
    move_input: bool = True,
) -> dict[str, Any]:
    """
    Move a legacy run (MyDrive/MD_output*) into projects/<dataset>/ and
    rewrite the absolute paths stored in its JSON/CSV/CFG files, so the
    finished steps are recognised (skipped) in the new location.

    A target folder must not exist yet or be empty. ``move_input=False``
    leaves MyDrive/MD_output where it is (then pass it as ``input_dir`` to
    make_layout). ``dry_run=True`` (default) only reports the plan.
    """
    old = make_layout(drive, dataset, mode="legacy")
    new = make_layout(drive, dataset, mode="project")

    attrs = (["input_dir"] if move_input else []) + list(_WORK_ATTRS)
    moves, problems = [], []
    for a in attrs:
        src, dst = getattr(old, a), getattr(new, a)
        if not src.is_dir():
            continue
        if dst.exists() and any(dst.iterdir()):
            problems.append(f"target not empty: {dst}")
        size, n = folder_size(src)
        moves.append({"from": str(src), "to": str(dst), "files": n, "size": _fmt(size)})

    report = {"dry_run": dry_run, "moves": moves, "problems": problems, "rewritten_files": 0}
    if not moves:
        report["problems"].append("no legacy MD_output* folders found")
    if dry_run or report["problems"]:
        return report

    for m in moves:
        dst = Path(m["to"])
        if dst.exists():
            dst.rmdir()                      # empty (checked above)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(m["from"], str(dst))

    mapping = [(m["from"], m["to"]) for m in moves]
    for root in [new.input_dir, *new.work_dirs]:
        report["rewritten_files"] += _rewrite_paths(root, mapping)
    mark_stage(new, "migration", moves=len(moves), rewritten_files=report["rewritten_files"])
    return report


if __name__ == "__main__":
    print(f"pipeline_layout.py {MODULE_VERSION}")
