#%% [markdown]
### Preparing a dataset


#%%
import pandas as pd
import numpy as np

df = pd.read_csv(
    "/Users/gaylonalfano/Code/python-statistical-analysis-course/section-2-exploring-data-analysis/Diabetes.csv"
)
df.info()

#%%
df.head()

#%% [markdown]
### `df.fillna(0)` returns a df with all nulls replaced with 0.
### `df.dropna()` also gives some good options to drop nulls

#%%
# Let's confirm that all null values in the df are replaced with a 0
df = df.dropna(axis=1)
df.info()


#%%
# We're interested in Glucose, BMI, and the Outcome column
df2 = df[["Glucose", "BMI", "Age", "Outcome"]]

#%%
df2.head()

#%%
# Discovered that we have some zeros for Glucose and BMI which isn't realistic
df2.describe()

#%%
# Replace the 0s with a null values. Let's exclude 'Outcome"
df2.columns[:-1]

#%%
# Check if any rows have a 0 value
df2[df2.columns[:-1]] == 0

#%% [markdown]
### Drop rows if any values are 0 (or True) by using `.any(axis=1)`

#%%
(df2[df2.columns[:-1]] == 0).any(axis=1)

#%% [markdown]
### Use the results of this to extract  the data we want


#%%
# My testing
df3 = df2[(df2[df2.columns[:-1]] == 0).any(axis=1) == False]
df3.info
df3.head()
#%% [markdown]
### Use `loc` and `~` (same as `not`)

#%%
df3 = df2.loc[~(df2[df2.columns[:-1]] == 0).any(axis=1)]
df3.describe()

#%%
df3.groupby("Outcome").mean()

#%% [markdown]
### Use `.agg()` function for more aggregate functions

#%%
# Link columns with agg methods with a dict
df3.groupby("Outcome").agg({"Glucose": "mean", "BMI": "median"})


#%%
# Run multiple agg methods on columns with a list
df3.groupby("Outcome").agg(["mean", "median"])


#%% [markdown]
### Split original df into two DFs.
# Useful if you want to run different methods on the separate DFs
# Need to use `.loc` to apply a mask/filter


#%%
positive = df3.loc[df3["Outcome"] == 1]
positive.info()

#%%
negative = df3.loc[df3["Outcome"] == 0]
negative.info()

#%%
print(positive.shape, negative.shape)

#%%
# Save to file using `.to_csv()`. Add index=False to not save indices
df3.to_csv(
    "/Users/gaylonalfano/Code/python-statistical-analysis-course/section-2-exploring-data-analysis/clean_diabetes.csv",
    index=False,
)

