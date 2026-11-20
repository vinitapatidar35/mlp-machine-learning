# Pandas Fundamentals: Notes

Machine Learning Practice (IIT Madras BS) | Week 1: Pandas and NumPy basics

A reference of the pandas and NumPy operations used for data loading, inspection, cleaning and summarising before any machine learning model is built. All examples use a small sample table (see section 1).

## Contents

- [Pandas Fundamentals: Notes](#pandas-fundamentals-notes)
  - [Contents](#contents)
  - [Key points](#key-points)
  - [1. Setup and sample data](#1-setup-and-sample-data)
  - [2. Series and DataFrame](#2-series-and-dataframe)
  - [3. Inspecting a dataset](#3-inspecting-a-dataset)
  - [4. Selecting data](#4-selecting-data)
    - [Columns](#columns)
    - [loc and iloc](#loc-and-iloc)
    - [Boolean filtering](#boolean-filtering)
  - [5. Adding and changing columns](#5-adding-and-changing-columns)
  - [6. Dropping data](#6-dropping-data)
  - [7. Missing values](#7-missing-values)
  - [8. Data types, dates and strings](#8-data-types-dates-and-strings)
  - [9. Sorting, unique values and counts](#9-sorting-unique-values-and-counts)
  - [10. Statistics](#10-statistics)
  - [11. GroupBy](#11-groupby)
  - [12. Combining tables](#12-combining-tables)
  - [13. Outliers (IQR rule)](#13-outliers-iqr-rule)
  - [14. Quick plotting](#14-quick-plotting)
  - [15. NumPy essentials](#15-numpy-essentials)
    - [The NumPy functions the exam actually uses](#the-numpy-functions-the-exam-actually-uses)
  - [16. Common pitfalls](#16-common-pitfalls)

## Key points

- `df.shape` is `(rows, columns)`. Number of features = `df.shape[1] - 1` when one column is the target.
- Count rows of a filtered table with `.shape[0]` or `len()`, not `.count()` (which counts per column).
- Combine conditions with `&` (and), `|` (or), `~` (not), each condition in brackets.
- Any comparison with NaN is `False`.
- `unique()` shows NaN, `value_counts()` skips it unless `dropna=False`.
- `loc` is label based with the end included. `iloc` is position based with the end excluded.
- Most methods return a new object. Reassign (`df = df.drop(...)`) to keep the change.

---

## 1. Setup and sample data

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")          # from a file
df = pd.read_csv("https://.../x.csv") # from a URL
df.to_csv("out.csv", index=False)     # save without the row numbers
```

Sample table used in the examples (`cltv` is the target, the rest are features):

```python
data = {
    "age":         [25, 32, 47, 51, 38, 29, 60, 44, 35, 52, 41, 28],
    "gender":      ["M","F","F","M","M","F","M","F","M","F","F","M"],
    "policy_type": ["Gold","Silver","Platinum","Bronze","Gold","Silver","Gold",np.nan,"Bronze","Platinum","Silver","Gold"],
    "income":      [30000,45000,90000,52000,np.nan,38000,75000,61000,28000,98000,47000,np.nan],
    "claims":      [0,1,2,0,1,0,3,1,0,2,1,0],
    "cltv":        [1200,2500,6800,1900,2300,1500,5200,3100,1100,7400,2600,1700],
}
df = pd.DataFrame(data)
```

---

## 2. Series and DataFrame

- **Series**: one labelled column (1D) with an index.
- **DataFrame**: a table (2D). Each column is a Series and all share the same index.

```python
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
s["b"]         # 20, by label
s.iloc[1]      # 20, by position
s.values       # underlying numpy array
s.index        # labels
s.dtype        # data type
s * 2          # element-wise operation
```

Creating a DataFrame:

```python
pd.DataFrame({"a": [1, 2], "b": [3, 4]})            # from a dict (keys = columns)
pd.DataFrame([[1, 3], [2, 4]], columns=["a", "b"])  # from a list of rows
pd.read_csv("file.csv")                             # from a file
```

---

## 3. Inspecting a dataset

| Code | Purpose |
|---|---|
| `df.head()` / `df.tail()` | first / last 5 rows |
| `df.shape` | (rows, columns) |
| `df.columns` | column names |
| `df.dtypes` | type of each column |
| `df.info()` | types, non-null counts, memory |
| `df.describe()` | summary of numeric columns |
| `df.describe(include="object")` | summary of text columns |
| `df.isna().sum()` | missing values per column |
| `df.nunique()` | distinct values per column |
| `df.duplicated().sum()` | number of duplicate rows |

`describe()` rows: `count, mean, std, min, 25%, 50%, 75%, max`. The `50%` row is the median.

Select columns by type:

```python
df.select_dtypes(include="number").columns   # numeric columns
df.select_dtypes(include="object").columns   # text / categorical columns
```

---

## 4. Selecting data

### Columns

```python
df["age"]                # Series
df[["age", "income"]]    # DataFrame (double brackets)
```

### loc and iloc

```python
df.loc[2]                      # row with label 2
df.loc[0:3]                    # rows 0..3 (end included, 4 rows)
df.loc[0:3, ["age", "cltv"]]   # rows and columns by label

df.iloc[0:3]                   # positions 0..2 (end excluded, 3 rows)
df.iloc[0:3, 0:2]              # first 3 rows, first 2 columns
df.iloc[-1]                    # last row
```

`loc` = label, `iloc` = integer position.

### Boolean filtering

```python
df[df["age"] > 40]
df[(df["age"] > 40) & (df["gender"] == "F")]     # and
df[(df["age"] > 50) | (df["claims"] >= 3)]       # or
df[~(df["gender"] == "M")]                       # not
df[df["policy_type"].isin(["Gold", "Silver"])]   # value in a list
df[df["age"].between(30, 40)]                    # inclusive on both ends
df[df["policy_type"].isna()]                     # missing values
df.query("age > 40 and gender == 'F'")           # SQL-like syntax
```

Counting and conditional update:

```python
df[df["age"] > 40].shape[0]
df.loc[df["age"] > 50, "claims"] = 0     # df.loc[mask, column] = value
```

---

## 5. Adding and changing columns

```python
df["income_k"] = df["income"] / 1000
df["age_group"] = df["age"].apply(lambda x: "senior" if x > 45 else "adult")
df["is_big"] = np.where(df["cltv"] > 3000, "yes", "no")
df["age_bin"] = pd.cut(df["age"], bins=[0, 30, 45, 100], labels=["young", "mid", "old"])
df = df.rename(columns={"claims": "num_claims", "income": "salary"})
```

| Tool | Use for |
|---|---|
| `Series.apply(func)` | custom rule on each value |
| `Series.map({...})` | replace values using a dictionary |
| `np.where(cond, a, b)` | two-way if/else |
| `pd.cut(col, bins, labels)` | numbers into groups |
| `df.apply(func, axis=1)` | rule that needs the whole row |

Note: with `lambda x: a if x > 45 else b`, a NaN goes to the else branch because `NaN > 45` is False.

---

## 6. Dropping data

```python
df2 = df.drop(columns=["claims"])            # one column
df2 = df.drop(columns=["claims", "income"])  # several columns
df2 = df.drop(index=[0, 1])                  # rows by label
df2 = df.dropna()                            # rows with any NaN
df2 = df.dropna(subset=["income"])           # only if income is NaN
df2 = df.drop_duplicates()
```

`drop` returns a new DataFrame and leaves `df` unchanged. `inplace=True` is being phased out, so reassign instead.

---

## 7. Missing values

```python
df.isna().sum()                  # per column
df.isna().sum().sum()            # total

df["income"] = df["income"].fillna(df["income"].mean())
df["income"] = df["income"].fillna(df["income"].median())
df["policy_type"] = df["policy_type"].fillna(df["policy_type"].mode()[0])
df["policy_type"] = df["policy_type"].fillna("Unknown")
```

`mode()` returns a Series (ties are possible), so take `[0]`. In sklearn the same job is done by `SimpleImputer` (fit on the training set only).

---

## 8. Data types, dates and strings

```python
df["age"] = df["age"].astype(float)
df["signup_date"] = pd.to_datetime(df["signup_date"])   # convert first
df["signup_date"].dt.year
df["signup_date"].dt.month
df["signup_date"].dt.day_name()
df["signup_date"].dt.dayofweek                          # 0 = Monday
(df["d2"] - df["d1"]).dt.days                           # days between dates
df[df["signup_date"].dt.year == 2024]
```

String methods (`.str`):

```python
df["gender"].str.lower()
df["policy_type"].str.strip()
df["policy_type"].str.contains("Go")
df["policy_type"].str.replace("Gold", "G")
df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})
```

`.dt` only works after `pd.to_datetime`.

---

## 9. Sorting, unique values and counts

```python
df.sort_values("age", ascending=False)
df.sort_values(["policy_type", "age"])
df.nlargest(3, "cltv")
df.nsmallest(3, "cltv")

df["policy_type"].unique()                      # includes NaN
df["policy_type"].nunique()                     # ignores NaN
df["policy_type"].value_counts()                # skips NaN
df["policy_type"].value_counts(dropna=False)
df["policy_type"].value_counts(normalize=True)  # proportions
```

---

## 10. Statistics

```python
df["age"].mean()   df["age"].median()   df["age"].std()
df["age"].min()    df["age"].max()      df["age"].sum()
df["age"].count()                 # non-missing values
df["age"].quantile(0.25)
df["age"].idxmax()                # index label of the maximum
df.mean(numeric_only=True)
df.corr(numeric_only=True)        # correlation matrix
df["age"].corr(df["cltv"])
df.cov(numeric_only=True)
```

Most statistics skip NaN automatically.

---

## 11. GroupBy

Split the rows into groups, apply a calculation to each group, combine the results.

```python
df.groupby("gender")["cltv"].mean()
df.groupby("gender")[["age", "cltv"]].mean()
df.groupby(["gender", "policy_type"])["cltv"].mean()
df.groupby("gender")["cltv"].agg(["mean", "max", "min"])
df.groupby("gender").agg(avg_cltv=("cltv", "mean"), max_age=("age", "max"))  # named aggregation
df.groupby("gender").size()
df.groupby("gender")["cltv"].transform("mean")   # group result repeated on every row
df.groupby("gender")["cltv"].mean().idxmax()     # group with the highest mean
```

Cross tables:

```python
pd.crosstab(df["gender"], df["policy_type"])
df.pivot_table(values="cltv", index="gender", columns="policy_type", aggfunc="mean")
```

`pivot()` needs every row/column pair to be unique, otherwise use `pivot_table()`. `melt()` converts wide to long.

---

## 12. Combining tables

```python
pd.concat([df1, df2], axis=0)                 # stack rows
pd.concat([df1, df2], axis=1)                 # side by side
pd.merge(df1, df2, on="id", how="inner")      # inner / left / right / outer
```

---

## 13. Outliers (IQR rule)

```python
Q1 = df["income"].quantile(0.25)
Q3 = df["income"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["income"] < lower) | (df["income"] > upper)]
outliers.shape[0]
```

---

## 14. Quick plotting

```python
import matplotlib.pyplot as plt
import seaborn as sns

df["age"].hist(bins=10); plt.show()
plt.scatter(df["age"], df["cltv"]); plt.show()
sns.boxplot(x=df["income"]); plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True); plt.show()
```

---

## 15. NumPy essentials

```python
a = np.array([3, 1, 2])
np.zeros((2, 3)); np.ones((2, 3)); np.arange(0, 10, 2); np.linspace(0, 1, 5)
a.shape; a.reshape(3, 1); a.T; a.ravel()
np.unique(a); np.where(a > 1)
a.argmax(); a.argmin(); a.argsort()          # positions, not values
np.vstack([a, a]); np.hstack([a, a])
np.random.seed(42); np.random.rand(3)
```

`argsort()` returns the positions that would sort the array. It is used later for ranking feature importances and finding nearest neighbours.

### The NumPy functions the exam actually uses

| Function | What it does | Example |
|---|---|---|
| `np.nan` | a missing value | `df.loc[df["RM"] < 1, "RM"] = np.nan` |
| `np.isnan(a)` | True where a value is missing | `np.isnan(a).sum()` |
| `np.where(condition)` | positions where the condition is True | `np.where(~rfe.support_)[0]` |
| `np.argmax(a)` / `np.argmin(a)` | position (starting at 0) of the largest or smallest value | `np.argmax(np.abs(model.coef_))` |
| `np.abs(a)` | absolute value (removes minus signs) | `np.abs(np.array([-3, 5, -1]))` gives `[3 5 1]` |
| `np.sqrt(a)` | square root | RMSE is `np.sqrt(mean_squared_error(y_test, pred))` |
| `np.round(x, 2)` | rounds to 2 decimals | `np.round(3.14159, 2)` |
| `np.mean(a)`, `a.mean()` | average of all values in the array | mean of a transformed matrix |
| `a.max()`, `a.min()`, `a.sum()` | largest, smallest, total | max of a transformed test matrix |
| `a.shape` | (rows, columns) | `X_train_t.shape[1]` is the number of features |

`argmax` and `argmin` give a position, not the value. For example, for `a = np.array([3, 9, 2, 7])`, `np.argmax(a)` is `1` and `np.argmin(a)` is `2`. The `~` sign flips True and False, so `~mask` means "not the mask".

If you see `NameError: name 'np' is not defined`, you forgot `import numpy as np`.

---

## 16. Common pitfalls

| Pitfall | Correct approach |
|---|---|
| Counting features from rows | `df.shape[1] - 1` |
| `df[cond].count()` for a row count | `df[cond].shape[0]` |
| `&` where "or" is meant | use `\|` |
| Swapped IQR fences | lower = Q1 - 1.5 IQR, upper = Q3 + 1.5 IQR |
| Reading the `50%` row as the mean | `50%` is the median |
| Expecting `drop` to change `df` | reassign the result |
| `.dt` on a text column | call `pd.to_datetime` first |
| NaN falling into the else branch of a lambda | fill or handle NaN first |
| Leaving a placeholder like `col` in code | use the real column name |

---

