import pandas as pd
import seaborn as sns
from  matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\PMB_MJU\dl_result\2103181727_timelapse_analyze\lrp_time3.csv", index_col=0)
sns.lineplot(data=df, x="time", y="stage", hue="treatment")
plt.show()
