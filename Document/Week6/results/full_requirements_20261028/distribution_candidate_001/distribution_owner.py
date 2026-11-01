from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path

import psutil

ROOT = Path(r"C:/Users/联想/Documents/ChatGPT/Intern/week6-worktree")
HERE = Path(__file__).resolve().parent
PYTHON = r"D:/Developer/multimodal-desktop-gui-agent/.venv-week5/Scripts/python.exe"
FREEZE = ROOT / "Document/Week6/results/full_requirements_20261028/semantic_caller_predicates_001/final_source_freeze_001.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    with (HERE / name).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def identity(pid):
    proc = psutil.Process(pid)
    return {"pid": proc.pid, "create_time": proc.create_time(), "exe": proc.exe(),
            "cmdline": proc.cmdline(), "parents": [
                {"pid": parent.pid, "create_time": parent.create_time(), "exe": parent.exe()}
                for parent in proc.parents()[:4]]}


BOOTSTRAP = r'''import json,os,runpy,sys
from pathlib import Path
import psutil
p=psutil.Process()
Path(sys.argv[1]).write_text(json.dumps({"pid":p.pid,"create_time":p.create_time(),"exe":p.exe(),"cmdline":p.cmdline(),"parents":[{"pid":x.pid,"create_time":x.create_time(),"exe":x.exe()} for x in p.parents()[:4]]},ensure_ascii=False,indent=2)+"\n",encoding="utf8")
sys.argv=["pip",*sys.argv[2:]]
runpy.run_module("pip",run_name="__main__")
'''


def run(name, arguments, *, cwd=HERE, pip=False):
    child_identity = HERE / f"{name}.actual_child_identity.json"
    command = ([PYTHON, "-X", "utf8", "-c", BOOTSTRAP, str(child_identity), *arguments]
               if pip else [PYTHON, "-I", "-X", "utf8", *arguments])
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    with (HERE / f"{name}.stdout.txt").open("wb") as stdout, (HERE / f"{name}.stderr.txt").open("wb") as stderr:
        child = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                                 env={**os.environ, "PYTHONUTF8": "1"})
        launcher = identity(child.pid)
        code = child.wait()
    receipt = {"name": name, "command": command, "effective_pip_arguments": arguments if pip else None,
               "cwd": str(cwd), "started_at_utc": started, "elapsed_seconds": time.monotonic() - clock,
               "launcher": launcher, "waited_launcher_pid": child.pid, "child_wait_exit_code": code,
               "actual_child_identity": json.loads(child_identity.read_text("utf8")) if child_identity.exists() else None}
    write(f"{name}.receipt.json", receipt)
    print(f"{name}: real child wait exit {code}", flush=True)
    if code != 0:
        raise RuntimeError(f"{name} failed with real child wait exit {code}; no dependency installation or retry")
    return receipt


def audit_wheel(wheel):
    expected = {str(path.relative_to(ROOT / "src")).replace("\\", "/"): path.read_bytes()
                for path in (ROOT / "src/gui_agent").rglob("*") if path.is_file()
                and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}}
    with zipfile.ZipFile(wheel) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        assert len(names) == len(set(names))
        assert archive.testzip() is None
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        modules = {name: archive.read(name) for name in names if name.startswith("gui_agent/")}
        assert set(modules) == set(expected), (set(modules) ^ set(expected))
        assert all(modules[name] == expected[name] for name in modules)
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        assert metadata["Name"] == "multimodal-desktop-gui-agent" and metadata["Version"] == "2.1.0"
        wheel_metadata_name = next(name for name in names if name.endswith(".dist-info/WHEEL"))
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_metadata_name))
        assert wheel_metadata["Root-Is-Purelib"] == "true" and wheel_metadata["Tag"] == "py3-none-any"
        record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
        record = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf8"))))
        assert len(record) == len(names) and {row[0] for row in record} == set(names)
        for name, digest, size in record:
            if name == record_name:
                assert digest == size == ""
            else:
                content = archive.read(name)
                encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
                assert digest == "sha256=" + encoded and size == str(len(content)), name
        result = {"wheel": str(wheel), "wheel_sha256": sha(wheel.read_bytes()), "wheel_bytes": wheel.stat().st_size,
                  "name": metadata["Name"], "version": metadata["Version"], "requires_python": metadata["Requires-Python"],
                  "package_file_count": len(modules), "package_python_module_count": sum(name.endswith(".py") for name in modules),
                  "source_package_file_set_equal": True, "all_source_package_bytes_equal": True,
                  "package_source_sha256": {name: sha(data) for name, data in expected.items()},
                  "all_record_hashes_and_sizes_valid": True, "record_entry_count": len(record), "crc_valid": True,
                  "has_console_entrypoints": any(name.endswith(".dist-info/entry_points.txt") for name in names),
                  "has_package_main_module": "gui_agent/__main__.py" in names,
                  "metadata_readme_is_pending_release_text": True}
    write("wheel_audit.json", result)
    return result


def main():
    owner = identity(os.getpid())
    write("owner_start.json", {"owner": owner, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                                "scope": "Private candidate wheel build and installed-source smoke; no clean dependency compatibility claim."})
    freeze = json.loads(FREEZE.read_text("utf8"))
    pins = freeze["code_pins"]

    def current():
        return {name: sha((ROOT / name).read_bytes()) for name in pins}

    before = current()
    assert before == pins and len(pins) == 256
    write("code_pins_before.json", before)
    readme_before = sha((ROOT / "README.md").read_bytes())
    write("build_plan.json", {"freeze_path": str(FREEZE), "freeze_sha256": sha(FREEZE.read_bytes()),
                               "code_pin_count": len(pins), "README_sha256": readme_before,
                               "pip_network_index_disabled": True, "dependency_installation_disabled": True,
                               "build_isolation_disabled": True, "version": "2.1.0"})
    results = []
    wheel_audit = None
    error = ""
    started = time.monotonic()
    try:
        wheel_dir = HERE / "wheel"
        wheel_dir.mkdir(exist_ok=False)
        results.append(run("build", ["wheel", "--no-index", "--no-deps", "--no-build-isolation", ".",
                                     "--wheel-dir", str(wheel_dir)], cwd=ROOT, pip=True))
        wheels = list(wheel_dir.glob("*.whl"))
        assert len(wheels) == 1
        wheel = wheels[0]
        wheel_audit = audit_wheel(wheel)
        target = HERE / "isolated_install"
        assert not target.exists()
        results.append(run("install", ["install", "--no-index", "--no-deps", "--no-compile", "--target", str(target),
                                       str(wheel)], pip=True))
        copies = HERE / "cli_sources"
        copies.mkdir(exist_ok=False)
        for name in ("week4_agent_cli.py", "week6_agent_cli.py"):
            (copies / name).write_bytes((ROOT / "scripts" / name).read_bytes())
        smoke = HERE / "smoke_installed.py"
        smoke.write_text(SMOKE, encoding="utf8")
        results.append(run("smoke", [str(smoke), str(target), str(copies), str(HERE)], cwd=HERE))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    after = current()
    write("code_pins_after.json", after)
    receipt = {"owner": owner, "results": results, "elapsed_seconds": time.monotonic() - started,
               "error": error, "candidate_preparation_succeeded": not bool(error), "code_pin_count": len(pins),
               "code_unchanged": before == after == pins, "README_unchanged": readme_before == sha((ROOT / "README.md").read_bytes()),
               "wheel_sha256": wheel_audit["wheel_sha256"] if wheel_audit else None,
               "new_model_gui_capture_input_calls": 0, "full_release_or_clean_dependency_validation": False,
               "limitations": ["Existing venv dependencies were reused; target isolates package files, not all dependencies.",
                               "Wheel contains no CLI entrypoint; help smoke uses exact frozen source CLI copies with installed wheel imports.",
                               "Wheel metadata embeds pending README; rebuild after final documentation changes for release.",
                               "Generated build/egg-info files are not source pins or source ZIP inputs."]}
    write("owner_receipt.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key not in {"owner", "results"}}, ensure_ascii=False), flush=True)
    return 0 if not error and receipt["code_unchanged"] and receipt["README_unchanged"] else 1


SMOKE = r'''import json,os,runpy,sys
from pathlib import Path
import psutil
target,copies,here=map(Path,sys.argv[1:])
p=psutil.Process()
(here/"smoke.actual_child_identity.json").write_text(json.dumps({"pid":p.pid,"create_time":p.create_time(),"exe":p.exe(),"cmdline":p.cmdline(),"parents":[{"pid":x.pid,"create_time":x.create_time(),"exe":x.exe()} for x in p.parents()[:4]]},ensure_ascii=False,indent=2)+"\n",encoding="utf8")
sys.dont_write_bytecode=True
sys.path[:0]=[str(target),str(copies)]
import gui_agent
from importlib.metadata import distribution
from gui_agent.planning.semantic_workflow import SEMANTIC_PROTOCOL_VERSION,SemanticDecomposer
from gui_agent.planning.workflow_decomposer import AutomaticWorkflowRunner,WorkflowRequest
assert gui_agent.__version__=="2.1.0"
assert SEMANTIC_PROTOCOL_VERSION=="week6.semantic_subtasks.v5"
assert Path(gui_agent.__file__).resolve().is_relative_to(target.resolve())
metadata=distribution("multimodal-desktop-gui-agent")
assert metadata.version=="2.1.0" and Path(metadata.locate_file("")).resolve()==target.resolve()
sys.argv=[str(copies/"week6_agent_cli.py"),"--help"]
code=None
try:
 runpy.run_path(str(copies/"week6_agent_cli.py"),run_name="__main__")
except SystemExit as exc:
 code=exc.code
assert code==0,code
loaded={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if (name=="gui_agent" or name.startswith("gui_agent.")) and getattr(module,"__file__",None)}
assert loaded and all(Path(path).is_relative_to(target.resolve()) for path in loaded.values())
(here/"smoke_audit.json").write_text(json.dumps({"package_version":gui_agent.__version__,"semantic_protocol":SEMANTIC_PROTOCOL_VERSION,"distribution_version":metadata.version,"loaded_package_files":loaded,"all_gui_agent_imports_from_isolated_target":True,"cli_help_exit_code":code,"cli_is_exact_source_copy_not_wheel_entrypoint":True,"existing_venv_dependencies_reused":True,"not_clean_dependency_compatibility_test":True,"no_model_gui_executor_constructed_or_called":True},ensure_ascii=False,indent=2)+"\n",encoding="utf8")
'''


if __name__ == "__main__":
    raise SystemExit(main())
