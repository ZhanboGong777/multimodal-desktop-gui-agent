import json,os,runpy,sys
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
