"""Only the independent grounding/telemetry CPU counterexamples; no model or GUI."""
from __future__ import annotations
import hashlib,json,os,subprocess,time,xml.etree.ElementTree as ET
from datetime import UTC,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent/"grounding_consistency_review_001/cpu_validation_003"
PYTHON=Path("D:/Developer/multimodal-desktop-gui-agent/.venv-week5/Scripts/python.exe")
PINS=["src/gui_agent/runtime/action_adapter.py","src/gui_agent/runtime/runner.py",
 "src/gui_agent/runtime/monitoring.py","src/gui_agent/runtime/visual_grounding.py",
 "src/gui_agent/planning/semantic_workflow.py","src/gui_agent/planning/workflow_decomposer.py",
 "src/gui_agent/planning/prompts.py","src/gui_agent/planning/planner.py",
 "src/gui_agent/runtime/schemas.py","src/gui_agent/runtime/verification.py",
 "tests/test_week6_grounding_consistency_review.py","tests/test_runtime_runner.py",
 "tests/test_week6_semantic_subtasks.py","tests/test_week6_automatic_decomposition.py",
 "tests/test_week6_rolling_actions.py","tests/test_week6_full_requirements_review.py",
 "scripts/week6_audit_semantic_acceptance.py"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 before={name:sha(ROOT/name) for name in PINS}
 for name in PINS:
  p=OUT/"source_snapshot"/(name+".txt");p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
 cmd=[str(PYTHON),"-m","pytest","-q","tests/test_week6_grounding_consistency_review.py",
  "--basetemp=.pytest-tmp-grounding-consistency-review003","--tb=short",f"--junitxml={OUT/'junit.xml'}"]
 start=datetime.now(UTC).isoformat();clock=time.perf_counter()
 with (OUT/"stdout.txt").open("wb") as stdout,(OUT/"stderr.txt").open("wb") as stderr:
  proc=subprocess.run(cmd,cwd=ROOT,env={**os.environ,"PYTHONUTF8":"1","PYTHONIOENCODING":"utf-8"},stdout=stdout,stderr=stderr,timeout=180,check=False)
 elapsed=time.perf_counter()-clock
 ruff_cmd=[str(PYTHON),"-m","ruff","check","tests/test_week6_grounding_consistency_review.py"]
 with (OUT/"ruff.stdout.txt").open("wb") as stdout,(OUT/"ruff.stderr.txt").open("wb") as stderr:
  ruff=subprocess.run(ruff_cmd,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=30,check=False)
 after={name:sha(ROOT/name) for name in PINS}
 cases=list(ET.parse(OUT/"junit.xml").getroot().iter("testcase"))
 failed=[c.attrib["name"] for c in cases if c.find("failure") is not None or c.find("error") is not None]
 result={"schema_version":"week6.grounding.independent_cpu_review.v1","reviewer":"/root/week6_replanning_review",
  "started_utc":start,"finished_utc":datetime.now(UTC).isoformat(),"command":cmd,"actual_returncode":proc.returncode,
  "wrapper_wall_seconds":elapsed,"ruff_command":ruff_cmd,"ruff_returncode":ruff.returncode,
  "collected_testcases":len(cases),"failed_testcases":failed,"selected_source_before":before,"selected_source_after":after,
  "selected_source_unchanged":before==after,"passed":proc.returncode==0 and ruff.returncode==0 and not failed and before==after,
  "operations":{"model_calls":0,"gui_operations":0,"full_pytest_runs":0},
  "scope":"Offline synthetic observations/model replies and CPU executors using real runtime. One pixel mapping is explicitly doubled. No real window success.",
  "source_runner_sha256":sha(Path(__file__)),
  "artifacts":{n:sha(OUT/n) for n in ("stdout.txt","stderr.txt","junit.xml","ruff.stdout.txt","ruff.stderr.txt")}}
 (OUT/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 print(json.dumps({"out":str(OUT),"returncode":proc.returncode,"cases":len(cases),"failures":failed,
  "pins_equal":before==after,"ruff":ruff.returncode,"audit_sha256":sha(OUT/"audit.json")},ensure_ascii=False))
 return 0 if result["passed"] else 1
if __name__=="__main__":raise SystemExit(main())

