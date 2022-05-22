import debugnn
import os

script = "scripts/visualize_plots.py"


def removesuffix(mystr: str):
  return int(str.removesuffix(mystr, ".pt"))

def othercfgsfunc(sub_dirs):
  othercfgs, n = [], len(sub_dirs)
  checkpoint_dirs = debugnn.append_basename(sub_dirs, "checkpoints")

  for i in range(n):
    try:
      checkpoint_files = os.listdir(checkpoint_dirs[i])
      minrloss = min(debugnn.maplist(checkpoint_files, removesuffix))
      othercfgs[i]["rloss"] = minrloss
    except FileNotFoundError:
      othercfgs[i]["rloss"] = None
  return othercfgs



def check_checkpoints(sub_dirs):
  sub_dirs_with_checkpoints = []
  for i in range(sub_dirs):
    if os.path.exists("{}/checkpoints".format(sub_dirs[i])):
      sub_dirs_with_checkpoints.append(sub_dirs[i])
  return sub_dirs_with_checkpoints



debugnn.run_scriptover(script, root="data/root", options="--test", othercfgsfunc=othercfgsfunc, filterfunc=check_checkpoints)
