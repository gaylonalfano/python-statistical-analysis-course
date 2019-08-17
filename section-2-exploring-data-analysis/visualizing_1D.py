#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn

print(os.getcwd())

#%%
d1 = np.loadtxt("./section-2-exploring-data-analysis/example_1.txt")
d2 = np.loadtxt("./section-2-exploring-data-analysis/example_2.txt")
print(d1.shape, d2.shape)

#%%
print(d1, d2)

#%% [markdown]
### Histogram Plots

#%%
plt.hist(d1, label="D1", bins=20)
plt.hist(d2, label="D2", bins=20)
plt.legend()
plt.ylabel("Counts")

#%%
# Unify the num of bins

counts1, bins, _ = plt.hist(d1, bins=10, label="D1")
plt.hist(d2, bins=bins, label="D2")
plt.legend()
plt.ylabel("Counts")


#%%
# Compute the range of values of both sets using np.linspace()
bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50)
counts1, _, _ = plt.hist(d1, bins=bins, label="D1")
plt.hist(d2, bins=bins, label="D2")
plt.legend()
plt.ylabel("Counts")

#%%
bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50)
counts1, _, _ = plt.hist(d1, bins=bins, label="D1", density=True)
plt.hist(d2, bins=bins, label="D2", density=True)
plt.legend()
plt.ylabel("Probability")

#%%
# Add histtype="step" for a better look
bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50)
plt.hist(
    [d1, d2], bins=bins, label="Stacked", density=True, histtype="barstacked", alpha=0.6
)
plt.hist(d1, bins=bins, label="D1", density=True, histtype="step", lw=1)
plt.hist(d2, bins=bins, label="D2", density=True, histtype="step", ls=":")
plt.legend()
plt.ylabel("Probability")

#%%
bins = 50
plt.hist(
    [d1, d2], bins=bins, label="Stacked", density=True, histtype="barstacked", alpha=0.6
)
plt.hist(d1, bins=bins, label="D1", density=True, histtype="step", lw=1)
plt.hist(d2, bins=bins, label="D2", density=True, histtype="step", ls=":")
plt.legend()
plt.ylabel("Probability")


#%% [markdown]
### Diabetes dataset
#### Trying to use this by for plotting our clean_diabetes.csv data
#### There are four columns

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./section-2-exploring-data-analysis/clean_diabetes.csv")
df.shape
df.head()

#%%
bins = 30
plt.hist(df["Glucose"], density=True, bins=bins, label="Glucose", histtype="step")
plt.hist(df["BMI"], density=True, bins=bins, label="BMI", histtype="step")
plt.hist(df["Age"], density=True, bins=bins, label="Age", histtype="step")
# plt.hist([df["Glucose"], df["BMI"], df["Age"]], bins=10, density=True, histtype="step")
plt.legend()


#%%
