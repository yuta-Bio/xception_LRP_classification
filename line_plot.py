import pandas as pd
import seaborn as sns
from  matplotlib import pyplot as plt
import os

fig, ax = plt.subplots(figsize = (10, 10))
plt.ylim(0, 7)
# plt.xlim(0, 1.)
path = r"C:\Users\PMB_MJU\dl_result\2107191542_timelapse_analyze\lrp_time3.csv"
df = pd.read_csv(path, index_col=0)
sns.lineplot(data=df, x="time", y="stage", hue="treatment")
sns.despine()
ax.tick_params(width=1.5)
dir = os.path.dirname(path)
plt.savefig(str(os.path.join(dir, "figure_1.png")))
plt.savefig(str(os.path.join(dir, "figure_1.svg")))

