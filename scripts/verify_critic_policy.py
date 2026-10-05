#!/usr/bin/env python3
"""Retain a compact current-source acceptance beside full historical raw ledgers."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from dongxi_llms.critic_policy_lab import run_reference
from dongxi_llms.run_identity import file_digest


def without_time(obj):
    if isinstance(obj,dict):return {k:without_time(v) for k,v in obj.items() if k not in ("seconds","wall_seconds")}
    if isinstance(obj,list):return [without_time(v) for v in obj]
    return obj


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--notebook-manifest",type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():parser.error("Unused report required; preserve historical source/evidence")
    notebook="notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb"
    files=["src/dongxi_llms/critic_policy_lab.py","tests/test_critic_policy_lab.py","scripts/verify_critic_policy.py",
        "src/dongxi_llms/char_reward_lab.py","src/dongxi_llms/text_reward_lab.py","src/dongxi_llms/run_identity.py",
        "fixtures/critic-policy/protocol.json","fixtures/critic-policy/preferences-balanced.json",
        "fixtures/critic-policy/preferences-confounded.json","fixtures/critic-policy/README.md",
        "experiments/specs/2026-10-04-critic-policy.md",notebook,"notebooks/day-20/README.md",
        "book/chapters/12-language-generation-as-a-policy.md","book/solutions/12-language-generation-as-a-policy.md",
        "book/labs/12-language-generation-as-a-policy.md"]
    files += [f"notebooks/figures/chapter-12/day-20-04_learned_critics_and_frozen_text_rewards-{i:02d}.png" for i in range(1,6)]
    before={p:file_digest(ROOT/p) for p in files}
    manifest=json.loads(args.notebook_manifest.read_text())
    row=next((r for r in manifest["notebooks"] if r["path"]==notebook),None)
    if manifest["failures"] or row is None or row["sha256"]!=before[notebook] or row["images"]!=5:
        parser.error("Matching passed fresh notebook with five figures required")
    workspace=Path(tempfile.mkdtemp(prefix="dongxi-critic-replay-"))
    began=time.perf_counter()
    actual=run_reference(ROOT,workspace/"frozen-rewards")
    old=json.loads(args.reference.read_text())["results"]
    comparisons=[]
    for current,previous in zip(actual["runs"],old["runs"]):
        comparison={"reward_arm":current["reward_arm"],"critic":current["critic"],"seed":current["seed"],
            "coordinates_match":[(current[k],previous[k]) for k in ("reward_arm","critic","seed")],
            "actor_state_match":current["actor_final_sha256"]==previous["actor_final_sha256"],
            "critic_state_match":current["critic_final_sha256"]==previous["critic_final_sha256"],
            "reward_state_match":current["reward_state_sha256"]==previous["reward_state_sha256"],
            "all_update_metrics_match":current["history"]==previous["history"],
            "all_raw_training_paths_match":current["training_ledger"]==previous["training_ledger"],
            "all_evaluation_records_match":without_time(current["panels"])==without_time(previous["panels"]),
            "frozen_control_match":without_time(current["frozen_control"])==without_time(previous["frozen_control"]),
            "cap4_match":without_time(current["cap4_intervention"])==without_time(previous["cap4_intervention"]),
            "actor_sha256":current["actor_final_sha256"],"critic_sha256":current["critic_final_sha256"],
            "reward_sha256":current["reward_state_sha256"]}
        comparisons.append(comparison)
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src"),CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    checks=[]
    for command in ([sys.executable,"-m","unittest","discover","-s","tests","-p","test_critic_policy_lab.py","-v"],
                    [sys.executable,"scripts/check_book_math.py"]):
        completed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
        checks.append({"command":command,"exit_code":completed.returncode,"stdout":completed.stdout,"stderr":completed.stderr})
    after={p:file_digest(ROOT/p) for p in files}
    numeric_ok=(len(comparisons)==12 and len(actual["runs"])==len(old["runs"]) and not actual["failures"]
                and all(all(c[k] for k in ("actor_state_match","critic_state_match","reward_state_match",
                        "all_update_metrics_match","all_raw_training_paths_match","all_evaluation_records_match",
                        "frozen_control_match","cap4_match")) and all(a==b for a,b in c["coordinates_match"]) for c in comparisons))
    data={"schema_version":1,"package":"DXI-10","status":"passed" if numeric_ok and before==after and all(c["exit_code"]==0 for c in checks) else "failed",
        "date_utc":datetime.now(timezone.utc).isoformat(),"command":list(sys.orig_argv),"source_sha256":before,
        "source_changes_during_run":[p for p in before if before[p]!=after[p]],
        "historical_raw_reference":{"path":str(args.reference),"sha256":file_digest(args.reference),
            "boundary":"Previous measured-source snapshot retained; current-source numeric replay excludes actual wall durations only"},
        "actual_replay_seconds":time.perf_counter()-began,"comparisons":comparisons,
        "replay_reward_exports":{a:v["export"] for a,v in actual["reward_fits"].items()},
        "replay_workspace":str(workspace),"all_reward_exports_reload_exact":all(f["reload_exact"] for f in actual["reward_fits"].values()),
        "training_paths":sum(r["sampled_paths"] for r in actual["runs"]),
        "valid_training_response_tokens":sum(r["generated_response_tokens"] for r in actual["runs"]),
        "actor_updates":sum(r["actor_updates"] for r in actual["runs"]),"critic_updates":sum(r["critic_updates"] for r in actual["runs"]),
        "notebook_manifest_sha256":file_digest(args.notebook_manifest),"notebook_verification":row,"checks":checks,
        "scope":"Actual tiny CPU actor/critic, saved learned text rewards, fixed-seed full replay and fresh notebook; no pretrained/GPU/full PPO/human quality claim",
        "hardening":"Pre-fit actual actor ID collision and duplicate/empty vocabulary rejection; original numeric ledgers unchanged",
        "limitations":actual["limits"]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x") as handle:handle.write(json.dumps(data,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(args.report),"status":data["status"],"replayed_arms":len(comparisons),"seconds":data["actual_replay_seconds"]}))
    if data["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":main()
