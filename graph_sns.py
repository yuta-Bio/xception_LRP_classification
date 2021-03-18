import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy

df = pd.read_csv(r"C:\Users\uemur\OneDrive\labs\report\mine\graduateThesis\word\lrp_time2.csv", header=0, index_col=0)
sns.catplot(data=df, x="stage", y="time", kind="box", hue="treatment")
plt.show()