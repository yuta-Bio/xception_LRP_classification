import os
import datetime

basename = datetime.datetime.now().strftime("%y%m%d%H%M")
path = os.path.join("/home/pmb-mju/dl_result", basename)
os.mkdir(path)