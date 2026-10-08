from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('shared_pilot',ROOT.parent/'teacher-v3/pilot.py')
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
pilot.ROOT=ROOT
if __name__=='__main__': pilot.main()
