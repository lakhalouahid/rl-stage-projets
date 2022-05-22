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



def check_checkpoints(sub_dirs):
  sub_dirs_with_checkpoints = []
  for i in range(len(sub_dirs)):
    if os.path.exists("{}/checkpoints".format(sub_dirs[i])):
      checkpoints_dir = "{}/checkpoints".format(sub_dirs[i])
      if len(os.listdir(checkpoints_dir)) > 0:
          sub_dirs_with_checkpoints.append(sub_dirs[i])
  print(sub_dirs_with_checkpoints)
  return sub_dirs_with_checkpoints



debugnn.run_scriptover(script, root="grid80x80", othercfgsfunc=othercfgsfunc, filterfunc=check_checkpoints, filterfields=['cmd', 'train-started'])

