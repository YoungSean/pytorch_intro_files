from pathlib import Path
import os
SIM_DIR = Path(
    "/data/lab/YangxiaoLu/physicsImages/nops8/outputnops512_updateddust")


fnames = os.listdir(SIM_DIR)
#print(len(fnames))
print(fnames[:5])