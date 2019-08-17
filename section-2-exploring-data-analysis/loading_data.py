#%% [markdown]
### Loading Data
# * Manually loading a file
# * Using `np.loadtxt`
# * Using `np.genfromtxt`
# * Using `pd.read_csv`
# * Using `pickle`

#%%
# Import our libraries
import numpy as np
import pickle
import pandas as pd
import os

filename = "load.csv"

#%%
print(os.getcwd())
os.chdir(
    "/Users/gaylonalfano/Code/python-statistical-analysis-course/section-2-exploring-data-analysis"
)

#%% [markdown]
### Manually
cols = None
data = []
with open(filename) as f:
    for line in f.readlines():
        vals = line.replace("\n", "").split(",")
        if cols is None:
            cols = vals
        else:
            data.append([float(x) for x in vals])

df0 = pd.DataFrame(data, columns=cols)
print(df0.dtypes)
df0.head()

#%% [markdown]
df1 = np.loadtxt(fname=filename, skiprows=1, delimiter=",")
print(df1.dtype)
print(df1[:5, :])


#%%
# Errors since `names=True` makes it a 1D array
df2 = np.genfromtxt(filename, delimiter=",", names=True, dtype=None)
print(df2[:5, :])

#%% [markdown]
# Use dtype=None to let Numpy infer the data types
# Use `names=True` to return a 1D array and use the first row as columns.

#%%
df2 = np.genfromtxt(filename, delimiter=",", names=True, dtype=None)
print(df2[:5])

#%%
print(df2.dtype)
print(df2["A"][:5])

#%%
d3 = pd.read_csv(filename)
print(d3.dtypes)
d3.head()

#%% [markdown]
### Loading a pickle file
# Need to open the file manually using `with open("filename", "rb")`
# You need to `rb` to read ***binary***
# Cool thing is you can package any Python objects into pickle files

#%%
with open(file="load_pickle.pickle", mode="rb") as f:
    d4 = pickle.load(f)
print(d4.dtypes)
d4.head()

#%%
