from __future__ import annotations

"""Standalone HADDOCK3 EMscoring runner.

Runs topoaa -> emscoring on already prepared AF3/HADDOCK-ready PDBs.
It does not perform docking or AIR generation.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from datetime import datetime, timezone
import csv, json, math, re, shutil, subprocess

MODULE_VERSION = "1.0.0"
DEFAULT_NCORES = 2
DEFAULT_NEMSTEPS = 50

@dataclass
class EMScoringResult:
    status: str
    prediction_id: str
    input_pdb: str
    output_directory: str
    run_directory: str
    config_file: str
    score_tsv: str
    minimized_pdb: str
    haddock_score: Optional[float]
    evdw: Optional[float]
    eelec: Optional[float]
    edesolv: Optional[float]
    eair: Optional[float]
    bsa: Optional[float]
    returncode: Optional[int]
    command: str
    message: str
    skipped: bool = False

def _now(): return datetime.now(timezone.utc).isoformat()
def _safe(s): return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
def _f(x):
    try:
        x=float(str(x).strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def _exe(name):
    x=shutil.which(name)
    if x: return x
    p=Path(name).expanduser()
    if p.is_file(): return str(p.resolve())
    raise FileNotFoundError(f"HADDOCK3 executable not found: {name}")

def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def validate_emscoring_pdb(pdb_path, expected_chains=("A","B")):
    p=Path(pdb_path).expanduser().resolve()
    if not p.is_file(): return {"status":"FAIL","reason":f"PDB not found: {p}"}
    atoms=0; bad=0; ca={c:0 for c in expected_chains}; residues={c:set() for c in expected_chains}
    try:
        with p.open(encoding="utf-8", errors="replace") as h:
            for line in h:
                if not line.startswith(("ATOM  ","HETATM")): continue
                atoms+=1; chain=line[21].strip()
                try: resid=int(line[22:26].strip())
                except ValueError: resid=None
                try:
                    xyz=[float(line[30:38]),float(line[38:46]),float(line[46:54])]
                    bad += not all(math.isfinite(v) for v in xyz)
                except (ValueError,IndexError): bad+=1
                if chain in ca:
                    ca[chain]+=1
                    if resid is not None: residues[chain].add(resid)
    except Exception as e: return {"status":"FAIL","reason":str(e)}
    if not atoms: return {"status":"FAIL","reason":"No ATOM/HETATM records found."}
    missing=[c for c,n in ca.items() if n==0]
    if missing: return {"status":"FAIL","reason":f"Missing chains: {missing}","chain_atoms":ca}
    if bad: return {"status":"FAIL","reason":f"Invalid coordinates in {bad} atoms."}
    return {"status":"PASS","atom_count":atoms,"chain_atoms":ca,"chain_residues":{c:len(residues[c]) for c in expected_chains}}

def generate_emscoring_config(input_pdb, run_directory, ncores=DEFAULT_NCORES, nemsteps=DEFAULT_NEMSTEPS):
    p=Path(input_pdb).expanduser().resolve(); r=Path(run_directory).expanduser().resolve()
    if not p.is_file(): raise FileNotFoundError(p)
    if ncores<1 or nemsteps<0: raise ValueError("ncores must be >=1 and nemsteps >=0")
    cfg=r.parent/"emscoring.cfg"; cfg.parent.mkdir(parents=True,exist_ok=True)
    cfg.write_text(f'''run_dir = "{str(r).replace(chr(92),chr(47))}"
mode = "local"
ncores = {int(ncores)}
molecules = ["{str(p).replace(chr(92),chr(47))}"]

[topoaa]

[emscoring]
nemsteps = {int(nemsteps)}
''', encoding="utf-8")
    return cfg

def _find_score(run):
    xs=list(Path(run).rglob("emscoring.tsv")) or list(Path(run).rglob("*emscoring*.tsv"))
    return sorted(xs)[0] if xs else None

def _find_pdb(run):
    xs=list(Path(run).rglob("*emscoring*.pdb"))
    return sorted(xs)[0] if xs else None

def _norm(s): return re.sub(r"[^a-z0-9]","",s.lower())
def _col(cols, names):
    d={_norm(c):c for c in cols}
    for n in names:
        if _norm(n) in d: return d[_norm(n)]
    return None

def parse_emscoring_tsv(score_tsv):
    p=Path(score_tsv)
    with p.open(encoding="utf-8", errors="replace") as h: rows=list(csv.DictReader(h,delimiter="\t"))
    if not rows: raise RuntimeError(f"No rows in {p}")
    row=rows[0]; cols=list(row)
    def v(names):
        c=_col(cols,names); return _f(row.get(c)) if c else None
    return {"raw":row,"haddock_score":v(["score","haddock_score"]),"evdw":v(["vdw","evdw"]),"eelec":v(["elec","eelec"]),"edesolv":v(["desolv","edesolv"]),"eair":v(["air","eair"]),"bsa":v(["bsa"])}

def run_emscoring(input_pdb, output_directory, prediction_id=None, ncores=DEFAULT_NCORES, nemsteps=DEFAULT_NEMSTEPS, haddock_executable="haddock3", skip_completed=True):
    p=Path(input_pdb).expanduser().resolve(); out=Path(output_directory).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    pid=prediction_id or p.stem; manifest=out/"emscoring_manifest.json"
    if skip_completed and manifest.is_file():
        try:
            m=json.loads(manifest.read_text()); s=Path(m.get("score_tsv",""))
            if m.get("status")=="SUCCESS" and m.get("input_pdb")==str(p) and int(m.get("ncores",-1))==ncores and int(m.get("nemsteps",-1))==nemsteps and s.is_file():
                z=parse_emscoring_tsv(s)
                return EMScoringResult("SKIPPED",pid,str(p),str(out),m.get("run_directory",""),m.get("config_file",""),str(s),m.get("minimized_pdb",""),z["haddock_score"],z["evdw"],z["eelec"],z["edesolv"],z["eair"],z["bsa"],0,"","Existing valid result; skipped.",True)
        except Exception: pass
    val=validate_emscoring_pdb(p)
    if val["status"]!="PASS":
        return EMScoringResult("INVALID_PDB",pid,str(p),str(out),"","","","",None,None,None,None,None,None,None,"",val["reason"])
    try: exe=_exe(haddock_executable)
    except Exception as e:
        return EMScoringResult("HADDOCK3_NOT_FOUND",pid,str(p),str(out),"","","","",None,None,None,None,None,None,None,"",str(e))
    run=out/"haddock3_emscoring_run"
    if run.exists(): shutil.rmtree(run)
    cfg=generate_emscoring_config(p,run,ncores,nemsteps); cmd=[exe,str(cfg)]; start=_now()
    try: proc=subprocess.run(cmd,cwd=str(out),capture_output=True,text=True,check=False)
    except Exception as e:
        return EMScoringResult("EXECUTION_ERROR",pid,str(p),str(out),str(run),str(cfg),"","",None,None,None,None,None,None,None," ".join(cmd),str(e))
    score=_find_score(run); mpdb=_find_pdb(run)
    if proc.returncode!=0 or score is None:
        _json(manifest,{"module_version":MODULE_VERSION,"status":"FAILED","prediction_id":pid,"input_pdb":str(p),"run_directory":str(run),"config_file":str(cfg),"score_tsv":str(score or ""),"ncores":ncores,"nemsteps":nemsteps,"started_utc":start,"finished_utc":_now(),"returncode":proc.returncode,"stderr_tail":proc.stderr[-4000:]})
        return EMScoringResult("EXECUTION_FAILED",pid,str(p),str(out),str(run),str(cfg),str(score or ""),str(mpdb or ""),None,None,None,None,None,None,proc.returncode," ".join(cmd),"HADDOCK3 failed or emscoring.tsv was not produced.")
    try: z=parse_emscoring_tsv(score)
    except Exception as e:
        return EMScoringResult("PARSE_FAILED",pid,str(p),str(out),str(run),str(cfg),str(score),str(mpdb or ""),None,None,None,None,None,None,proc.returncode," ".join(cmd),str(e))
    _json(manifest,{"module_version":MODULE_VERSION,"status":"SUCCESS","prediction_id":pid,"input_pdb":str(p),"run_directory":str(run),"config_file":str(cfg),"score_tsv":str(score),"minimized_pdb":str(mpdb or ""),"ncores":ncores,"nemsteps":nemsteps,"started_utc":start,"finished_utc":_now(),"returncode":proc.returncode,"scores":{k:z[k] for k in ("haddock_score","evdw","eelec","edesolv","eair","bsa")}})
    return EMScoringResult("SUCCESS",pid,str(p),str(out),str(run),str(cfg),str(score),str(mpdb or ""),z["haddock_score"],z["evdw"],z["eelec"],z["edesolv"],z["eair"],z["bsa"],proc.returncode," ".join(cmd),"HADDOCK3 EMscoring completed successfully.")

def _record(x):
    if isinstance(x,Mapping):
        pid=x.get("prediction_id") or x.get("id") or x.get("fold_id")
        pdb=x.get("haddock_ready_pdb") or x.get("pdb_path") or x.get("input_pdb") or x.get("structure_pdb")
        if not pid or not pdb: raise KeyError(f"Invalid record: {x}")
        return str(pid),Path(pdb)
    if isinstance(x,(tuple,list)) and len(x)>=2: return str(x[0]),Path(x[1])
    raise TypeError("Each record must be a mapping or (prediction_id, pdb_path) pair.")

def _csv(path, results):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not results: return
    rows=[asdict(x) for x in results]
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def batch_emscoring(records, output_root, ncores=DEFAULT_NCORES, nemsteps=DEFAULT_NEMSTEPS, haddock_executable="haddock3", skip_completed=True, results_csv=None):
    root=Path(output_root).expanduser().resolve(); root.mkdir(parents=True,exist_ok=True)
    csv_path=Path(results_csv).expanduser().resolve() if results_csv else root/"haddock_emscoring_results.csv"
    results=[]
    for i,record in enumerate(records,1):
        pid,pdb=_record(record); out=root/_safe(pid)
        print("\n"+"="*80); print(f"HADDOCK EMscoring {i}"); print("="*80); print("Prediction ID:",pid); print("Input PDB:",pdb)
        r=run_emscoring(pdb,out,pid,ncores,nemsteps,haddock_executable,skip_completed); results.append(r); print("Status:",r.status)
        if r.haddock_score is not None: print("HADDOCK score:",r.haddock_score)
        _csv(csv_path,results)
    return results

def summarize_emscoring(results):
    d={"total":0,"success":0,"skipped":0,"invalid_pdb":0,"haddock3_not_found":0,"execution_failed":0,"parse_failed":0,"other_failed":0}
    for r in results:
        d["total"]+=1
        key={"SUCCESS":"success","SKIPPED":"skipped","INVALID_PDB":"invalid_pdb","HADDOCK3_NOT_FOUND":"haddock3_not_found","EXECUTION_FAILED":"execution_failed","PARSE_FAILED":"parse_failed"}.get(r.status,"other_failed")
        d[key]+=1
    return d

if __name__ == "__main__":
    print(f"haddock_emscoring.py {MODULE_VERSION}")
