import debugnn
import os

script = "scripts/visualize_plots.py"


def removesuffix(mystr: str):
  return int(float(str.removesuffix(mystr, ".pt")))

def othercfgsfunc(sub_dirs):
  othercfgs, n = [], len(sub_dirs)
  checkpoint_dirs = debugnn.append_basename(sub_dirs, "checkpoints")
  for i in range(n):
    checkpoint_files = os.listdir(checkpoint_dirs[i])
    minrloss = min(debugnn.maplist(checkpoint_files, removesuffix))
    othercfgs.append({})
    othercfgs[i]["rloss"] = minrloss
  return othercfgs




debugnn.run_scriptover(script, root="data/root", options="--test", othercfgsfunc=othercfgsfunc)
