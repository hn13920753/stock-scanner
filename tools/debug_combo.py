import runpy, sys
sys.argv = ["x"]
g = runpy.run_path("tools/research_combo.py", run_name="dbg")
