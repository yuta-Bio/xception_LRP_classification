import os
import glob
import datetime
import shutil
import re
import matplotlib.pyplot as plt
import pandas as pd


base_dir = ('/home/pmb-mju/dl_result')
basename = datetime.datetime.now().strftime("%y%m%d%H%M")
path = os.path.join(base_dir, basename)
os.mkdir(path)
shutil.copyfile(__file__, str(os.path.join(path, os.path.basename(__file__))))
shutil.copyfile('Image_data_generater_LRP.py', str(os.path.join(path, 'Image_data_generater_LRP.py')))


ls_true_ratio = []
ref_path = ('')
match1_path = ('')
ls_match_path = [match1_path]
for match_path in ls_match_path:
    ls_path = glob.glob(match_path + '/**/*.tif', recursive = True)
    num_true = 0
    num_false = 0
    for path in ls_path:
        stage_match = re.search('stage\d+', path).group()
        base_name = str(os.path.basename(path))
        match_ref_path = glob.glob(ref_path + '/**/' + base_name, recursive = True)[0]
        stage_ref = re.search('stage/d+', match_ref_path)
        if (stage_match == stage_ref):
            num_true += 1
        else:
            num_false += 1

    total_num = len(ls_path)
    true_ratio = num_true / total_num
    ls_true_ratio.append(true_ratio)

ax = plt.figure().add_axese([0,0,1,1])
ax.bar(list(range(len(ls_true_ratio))), ls_true_ratio, color = 'blue')
plt.savefig(str(os.path.join(path, 'human_classification.png')))
df = pd.DataFrame(ls_true_ratio)
df.to_csv(str(os.path.join(path, 'human_classification.csv')))
