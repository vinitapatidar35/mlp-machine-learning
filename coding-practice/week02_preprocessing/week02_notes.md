# Week 2: Preprocessing with scikit-learn

Complete notes, written from zero. Useful for OPPE 1 and also for the end term.

## Contents

1. The big picture
2. scikit-learn basics: imports and the tool map
3. X and y
4. Train and test split
5. fit, transform, predict and data leakage
6. Sample data used in this file
7. Missing values: SimpleImputer
8. Text to numbers: OrdinalEncoder and OneHotEncoder
9. Scaling: StandardScaler and MinMaxScaler
10. Hidden missing values (-1, ?, UNKNOWN)
11. ColumnTransformer
12. Pipeline and make_pipeline
13. The two-stage route (no Pipeline)
14. Feature selection: SelectKBest, RFE, SequentialFeatureSelector
15. PCA
16. Getting help: help(), ? and Ctrl+Space
17. Exam question to code table
18. Counting features after preprocessing
19. Reading errors
20. Common pitfalls
21. Practice exercises
22. End-term quick revision

---

## 1. The big picture

Raw data is messy. A model needs clean numbers. Preprocessing is everything we do between the raw table and the model.

| Problem in the raw data | Fix | Tool |
|---|---|---|
| Missing values (NaN) | fill them in (impute) | `SimpleImputer` |
| Text columns | turn into numbers (encode) | `OrdinalEncoder`, `OneHotEncoder` |
| Columns on very different scales | rescale | `StandardScaler`, `MinMaxScaler` |
| Too many columns | keep the useful ones, or combine them | `SelectKBest`, `RFE`, `PCA` |

The standard flow:

```
raw table -> hidden missing values to NaN -> split into train and test
          -> impute -> encode -> scale (all inside one ColumnTransformer)
          -> (optional) select features or PCA -> model
```

---

## 2. scikit-learn basics: imports and the tool map

scikit-learn (imported as `sklearn`) is a library of machine learning tools. It is big, so it is split into sections called modules. We import only the tool we need, using this rule:

```python
from sklearn.<module> import <Tool>
```

If you forget which module a tool lives in, look at the sklearn quick reference sheet (allowed in the exam) or type `sklearn.` and press `Ctrl + Space`.

| Module | What lives there | Tools we use |
|---|---|---|
| `sklearn.model_selection` | testing and choosing models | `train_test_split`, `GridSearchCV`, `KFold`, `cross_val_score` |
| `sklearn.impute` | missing values | `SimpleImputer` |
| `sklearn.preprocessing` | scaling, encoding | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `OrdinalEncoder`, `PolynomialFeatures` |
| `sklearn.compose` | applying different steps to different columns | `ColumnTransformer` |
| `sklearn.pipeline` | chaining steps | `Pipeline`, `make_pipeline` |
| `sklearn.feature_selection` | choosing columns | `SelectKBest`, `f_regression`, `RFE`, `SequentialFeatureSelector` |
| `sklearn.decomposition` | combining columns | `PCA` |
| `sklearn.linear_model` | linear models | `LinearRegression`, `Ridge`, `Lasso`, `SGDRegressor` |
| `sklearn.metrics` | scores | `r2_score`, `mean_absolute_error`, `mean_squared_error` |

---

## 3. X and y

- `y` is the target: the column we want to predict.
- `X` is everything else: the inputs (also called features).

```python
X = df.drop(columns=["cltv"])   # all columns except the target
y = df["cltv"]                  # the target column
```

Capital X (a table) and lowercase y (a single column) is the usual convention. If the target is the last column of the file, you can write `X = df.iloc[:, :-1]` and `y = df.iloc[:, -1]`.

---

## 4. Train and test split

We keep part of the data hidden from the model. The model learns on the training part. We judge it on the test part, which is like a surprise exam.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

- It returns exactly 4 things, always in this order: `X_train, X_test, y_train, y_test`.
- `test_size=0.3` means 30% of the rows go to the test set. The size rounds up when the number is not whole: 12 rows and 30% is 3.6, so the test set gets 4 rows and the train set gets 8.
- The default `test_size` is 0.25. The default `shuffle` is True (rows are mixed before splitting).
- `random_state` fixes the shuffle so you get the same split every time. Copy the number the question gives you exactly. A different number gives a different split and different answers.
- `stratify=y` keeps the class proportions the same in both sets. It is used for classification (OPPE 2) and sometimes in preprocessing questions that ask for it.

---

## 5. fit, transform, predict and data leakage

Every sklearn tool follows one pattern.

| Method | What it does | Used by |
|---|---|---|
| `fit(data)` | learns something from the data (a mean, a min, a list of categories, coefficients) | every tool |
| `transform(data)` | applies what was learned and returns the new data | transformers (imputers, encoders, scalers) |
| `fit_transform(data)` | `fit` then `transform` in one call | transformers |
| `predict(data)` | gives predictions | models |
| `score(X, y)` | gives R² (for regression models) | models |

The learned values are stored in attributes that end with an underscore: `statistics_`, `categories_`, `mean_`, `scale_`, `data_min_`, `data_max_`, `coef_`, `best_params_`.

The golden rule: **fit on train only. Transform both train and test.**

```python
scaler.fit(X_train)                     # learn from train only
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)    # use the numbers learned from train
```

Why: if the test data leaks into the learning (for example, its mean is used), the test is no longer a fair surprise exam. This mistake is called data leakage. For the same reason, never call `fit` or `fit_transform` on the test data.

---

## 6. Sample data used in this file

```python
import numpy as np
import pandas as pd

data = {
    "age":         [25, 32, 47, 51, 38, 29, 60, 44, 35, 52, 41, 28],
    "gender":      ["M","F","F","M","M","F","M","F","M","F","F","M"],
    "policy_type": ["Gold","Silver","Platinum","Bronze","Gold","Silver","Gold",np.nan,"Bronze","Platinum","Silver","Gold"],
    "income":      [30000,45000,90000,52000,np.nan,38000,75000,61000,28000,98000,47000,np.nan],
    "claims":      [0,1,2,0,1,0,3,1,0,2,1,0],
    "cltv":        [1200,2500,6800,1900,2300,1500,5200,3100,1100,7400,2600,1700],
}
df = pd.DataFrame(data)

X = df.drop(columns=["cltv"])
y = df["cltv"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape)   # (8, 5) (4, 5)
```

---

## 7. Missing values: SimpleImputer

A model cannot handle NaN. `SimpleImputer` fills each NaN with a value learned from train.

```python
from sklearn.impute import SimpleImputer

imp_income = SimpleImputer(strategy="median")
imp_income.fit(X_train[["income"]])
X_train[["income"]] = imp_income.transform(X_train[["income"]])
X_test[["income"]]  = imp_income.transform(X_test[["income"]])
print(imp_income.statistics_)       # the value used to fill
```

| `strategy` | Fills with | Works on |
|---|---|---|
| `"mean"` | average | numbers |
| `"median"` | middle value (safer with outliers) | numbers |
| `"most_frequent"` | the most common value | numbers and text |
| `"constant"` with `fill_value=0` | a fixed value you choose | numbers and text |

Important details:

- Double brackets: `X_train[["income"]]`. sklearn wants a 2D table (rows and columns), not a single column. Use double brackets on both sides of the `=`.
- Use a separate variable name for each imputer (`imp_income`, `imp_policy`). Reusing one name means the variable holds only the last imputer you made.
- The imputer must be fitted on train and used on both.
- The learned fill values are in `statistics_`, in the order of the columns you gave.

---

## 8. Text to numbers: OrdinalEncoder and OneHotEncoder

A model needs numbers. The choice depends on one question: does the text have a natural order?

| Has a natural order (Low, Medium, High) | No natural order (Red, Blue, Green) |
|---|---|
| `OrdinalEncoder` with your own order | `OneHotEncoder` |
| 1 column in, 1 column out | 1 column in, one column per unique word out |

If we gave Red, Blue and Green the numbers 0, 1 and 2, the model would think Green is bigger than Red. One-hot avoids that.

### OrdinalEncoder

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[["Bronze", "Silver", "Gold", "Platinum"]])
oe.fit(X_train[["policy_type"]])
codes = oe.transform(X_train[["policy_type"]])
print(oe.categories_)
```

- Without `categories=`, the default order is alphabetical (Bronze, Gold, Platinum, Silver). That is wrong for ordered data like this. Always give the order, lowest to highest.
- The double brackets in `categories=[[...]]` are needed because the setting takes one list per column.
- It cannot handle NaN in the older sense: fill the NaN first (for example with `most_frequent`).
- For words that appear in test but never in train, use `handle_unknown="use_encoded_value", unknown_value=-1`.
  (`handle_unknown="ignore"` is only for `OneHotEncoder`.)

### OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

oh = OneHotEncoder(sparse_output=False)
oh.fit(X_train[["gender"]])
dummies = oh.transform(X_train[["gender"]])
print(oh.categories_)                  # the words, in column order (alphabetical)
print(oh.get_feature_names_out())      # ['gender_F' 'gender_M']
```

- One new column per unique word learned from train. Gender (F, M) gives 2 columns. Policy type with 4 words gives 4.
- The new columns are in alphabetical order of the words.
- `sparse_output=False` gives a normal table instead of a compressed one.
- `handle_unknown="ignore"`: a word that appears only in test becomes all zeros instead of an error.
- `drop="first"` removes the first column of each feature, because it can be worked out from the others (if `gender_F` is 0, the person is M). Use it only if the question says so. Then the column count is (unique words minus 1).
- The encoder itself does not change your table. It only returns a new array of 0 and 1. The original column stays in your table until you drop it, or until a ColumnTransformer replaces it for you.

---

## 9. Scaling: StandardScaler and MinMaxScaler

Columns on very different scales (age about 40, income about 50000) are unfair to many models: the bigger numbers dominate. Scaling puts columns on a similar scale.

| Scaler | Formula | Result | Learned attributes |
|---|---|---|---|
| `StandardScaler` | (value minus mean) divided by standard deviation | mean about 0, standard deviation 1, can be negative | `mean_`, `scale_` |
| `MinMaxScaler` | (value minus min) divided by (max minus min) | train values in the range 0 to 1 | `data_min_`, `data_max_` |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

mm = MinMaxScaler()
mm.fit(X_train[["age"]])
print(mm.data_min_, mm.data_max_)
scaled_age = mm.transform(X_train[["age"]])

ss = StandardScaler()
ss.fit(X_train[["income"]])
print(ss.mean_, ss.scale_)
```

Key facts:

- Fit on train only. Test values use the train min, max, mean and standard deviation.
- A test value larger than the train max becomes more than 1 with MinMaxScaler. A test value smaller than the train min becomes negative. This is normal, not an error.
  Example: train min 28 and max 60. A test age of 20 becomes (20 minus 28) divided by 32, which is -0.25. A test age of 70 becomes 1.3125.
- A scaler does not fill NaN. It passes NaN through. Impute first.
- `scale_` is the standard deviation of the train column.
- `MinMaxScaler(feature_range=(-1, 1))` changes the target range.
- Do not run a scaling cell twice on the same column when you assign the result back. You would scale already-scaled numbers. If `mean_` is about 0 and `scale_` is 1 before you scaled anything yourself, you have scaled twice.

---

## 10. Hidden missing values (-1, ?, UNKNOWN)

Real data often hides missing values. They are not NaN, but they are impossible values:

- number of rooms equal to -1
- age equal to -2
- a text column with `"UNKNOWN"` or `"?"`

The exam hints: "carefully look at the values the feature takes and find the implausible value." The routine:

```python
print(sorted(df["RM"].unique()))
print(df["RIVERSIDE"].value_counts())

rm_bad  = df["RM"] < 1
age_bad = df["AGE"] < 0
riv_bad = df["RIVERSIDE"] == "UNKNOWN"
print(rm_bad.sum(), age_bad.sum(), riv_bad.sum())     # counts of hidden missing values

df.loc[rm_bad, "RM"] = np.nan                         # convert them to real NaN
df.loc[age_bad, "AGE"] = np.nan
df.loc[riv_bad, "RIVERSIDE"] = np.nan
```

Do this before the split if the question says to replace them with `np.nan` first. For a "?" in text, use `df == "?"` or `.replace("?", np.nan)`.

---

## 11. ColumnTransformer

It applies a different tool to each column, all in one go, and returns one final table. Without it, you would fit and transform each column by hand.

```python
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([
    ("age_mm",    MinMaxScaler(),                    ["age"]),
    ("income_ss", StandardScaler(),                  ["income"]),
    ("gender_oh", OneHotEncoder(sparse_output=False), ["gender"]),
], remainder="passthrough")

X_train_t = ct.fit_transform(X_train)
X_test_t  = ct.transform(X_test)
print(X_train_t.shape)
```

How to read it. The big square bracket holds a list of groups. Each group is a round bracket with three parts:

```
( "any label you choose",  Tool(),  ["real column name"] )
```

- The label is just a name for you. It can be anything.
- The tool is the one from the earlier sections.
- The column list must use the real column names, spelled exactly like the table. It is always a list, even for one column.

Rules:

- Columns you do not mention are dropped from the output. `remainder="passthrough"` keeps them, placed at the end, untouched.
- The output columns come in the order of your groups. One-hot columns are placed where the group is.
- Train: `fit_transform`. Test: `transform` only.
- The output is a NumPy array. Use `.shape`, `.mean()`, `.max()`, and `ct.get_feature_names_out()` for the column names. The names look like `age_mm__age` (group label, two underscores, column).
- If a question says "preprocess in exactly this order", write the groups in exactly that order. The shape and the mean of the whole matrix do not depend on the order, but "the value in column index k" does.
- One tool can be used in several groups ("you may have to use multiple instances of a transformer"). You can also give one group several columns: `("mm", MinMaxScaler(), ["DIS", "TAX", "BPL"])`.

---

## 12. Pipeline and make_pipeline

A Pipeline is a list of steps that run in order, as one tool. The output of one step goes into the next.

```
raw column -> [ imputer ] -> filled column -> [ scaler ] -> final column
```

You need it when one column needs two steps in a row, for example "impute with the median, then standard scale".

Two ways to write the same thing:

```python
from sklearn.pipeline import Pipeline, make_pipeline

# long form: you name the steps
p_long = Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", MinMaxScaler())])

# short form: names are made for you (lowercase class names)
p_short = make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler())
```

- Both give the same output. `make_pipeline` has fewer brackets, so fewer mistakes.
- The pipeline never chooses columns. It works on whatever data you give it, either directly (`p_short.fit_transform(df[["marks"]])`) or through a ColumnTransformer group.
- The first tool in the list runs first. An imputer must come before a scaler, because scalers leave NaN as NaN.
- `fit` fits every step in order on train. `transform` applies every step in order.

A pipeline inside a ColumnTransformer group (the exam pattern, for example "RM: impute with the mean, then min max scaling"):

```python
ct = ColumnTransformer([
    ("rm", make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler()), ["RM"]),
    ("riv", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(sparse_output=False)), ["RIVERSIDE"]),
])
```

The template to remember: ColumnTransformer, then a list of groups. Each group is (label, tool, [columns]). If a column needs two steps, the tool is a pipeline of the steps.

With `make_pipeline`, step names are the lowercase class names: `simpleimputer`, `minmaxscaler`, `pca`, `lasso`. If a question needs a step by name (for example `named_steps["pca"]`), use the right name. With `Pipeline`, you choose the names yourself.

---

## 13. The two-stage route (no Pipeline)

If Pipeline feels hard, split the work into two stages. The final numbers are the same, as long as you keep the order the question gives.

1. Impute first, as separate steps (fit on train, transform train and test).
2. Then use one ColumnTransformer with one tool per group (scale, encode).

The shape and the mean of the whole matrix are the same as with pipelines.

---

## 14. Feature selection

Some columns do not help predict the target. Feature selection keeps only the useful ones. All of these work on numbers only, with no NaN. Run them after preprocessing.

| Tool | Idea | Settings | Read the result |
|---|---|---|---|
| `SelectKBest` | score each column against the target, keep the top K | `score_func=f_regression`, `k=4` | `scores_`, `pvalues_`, `get_support()` |
| `RFE` | train a model, remove the weakest column, repeat until the number you asked for remains | the model, `n_features_to_select` | `support_`, `ranking_` |
| `SequentialFeatureSelector` | add (forward) or remove (backward) one column at a time, checking the model each time | the model, `n_features_to_select`, `direction`, `cv` | `get_support()` |

```python
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# SelectKBest: the selector is created without data. Data goes in at fit, and it needs X and y.
kb = SelectKBest(score_func=f_regression, k=2)
kb.fit(Xs, ys)
print(kb.scores_.round(1))        # one F-score per column, in column order
print(kb.pvalues_.round(4))       # one p-value per column
print(kb.get_support())           # True for kept columns
print(kb.transform(Xs).shape)     # (rows, 2)

# RFE
rfe = RFE(LinearRegression(), n_features_to_select=3)
rfe.fit(Xs, ys)
print(rfe.support_)               # True = kept
print(rfe.ranking_)               # 1 = kept; larger = removed earlier
print(np.where(~rfe.support_)[0]) # positions of the eliminated columns

# SequentialFeatureSelector
sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=3,
                                direction="forward", cv=KFold(5, shuffle=True, random_state=42))
sfs.fit(Xs, ys)
print(sfs.get_support())
```

Key points:

- `f_regression` is the judge (it scores). `SelectKBest` is the teacher (it picks). For a regression target, you always use `f_regression`. Other scoring functions (`f_classif`, `chi2`) are for classification.
- A higher F-score means a stronger link to the target. A p-value below 0.05 means the link is reliable.
- "How many features have p-value below 0.05": `(kb.pvalues_ < 0.05).sum()`.
- "Index of the highest F-score": `np.argmax(kb.scores_)`. Positions start at 0.
- "Which features does RFE eliminate when removing 2": `n_features_to_select = total columns - 2`, then `np.where(~rfe.support_)[0]`.
- If a question says "eliminate k features", you keep the total minus k.

---

## 15. PCA

PCA does not choose columns. It creates new columns by mixing the old ones. The first new column (the first principal component) holds the most information, the second holds the next most, and so on. The new columns have no names like `age`.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(Xs)
print(pca.explained_variance_ratio_)      # share of information in each component
print(pca.explained_variance_ratio_[0])   # the first component's share
print(pca.explained_variance_ratio_.sum())

pca2 = PCA(n_components=0.95)             # keep enough components to hold 95%
pca2.fit(Xs)
print(pca2.n_components_)                 # how many were kept
print(pca2.transform(Xs).shape)
```

- `n_components=3` gives exactly 3 columns. `n_components=0.95` keeps as many as needed to hold 95% of the information. It can never be more than the number of columns.
- `explained_variance_ratio_` is a list with one number per component. The shares add up to at most 1. Index `[0]` is the first component.
- PCA works on numbers only, so text columns must be encoded first. If you pass text, you get `ValueError: could not convert string to float: 'F'`.
- `whiten=True` rescales the components to equal spread. `svd_solver="full"` chooses the method. Use them when the question lists them.
- Many questions apply PCA only when the data is already numeric and clean. Follow the order the question gives.

---

## 16. Getting help: help(), ? and Ctrl+Space

Allowed in the exam: `help()`, `?`, the autocomplete menu, and the sklearn quick reference sheet. Not allowed: other tabs (only SEEK, Colab and the sheet).

```python
from sklearn.preprocessing import MinMaxScaler
help(MinMaxScaler)     # prints the manual under the cell
?MinMaxScaler          # opens the same manual in a side panel (Colab)
```

Every help page has the same layout. You only need three parts.

| Part | What it holds | When to read it |
|---|---|---|
| Parameters | the settings you give when creating the tool. Each entry looks like `name : type, default=value` | when you need the exact spelling or the allowed options |
| Attributes | what the tool learned after `fit` (names end in `_`) | when a question asks for a learned value |
| Examples | lines starting with `>>>` that are working code | when you do not know how to write the tool at all |

Reading a line: `clip : bool, default=False` means the setting is called `clip`, the type is True or False, and the default is False.

A shortcut to jump to one part of a long help page:

```python
doc = MinMaxScaler.__doc__
i = doc.find("Attributes")
print(doc[i:i+800])
```

Ctrl + Space shows an autocomplete menu: type the first letters, or type a variable name and a dot (`mm.`), then press Ctrl + Space. It shows names only, so use `?` or `help()` for meanings. It works only after you import the tool.

Routine: import the tool, run `help(Tool)`, read Parameters for the settings and Attributes for the answer, write your code, run it, check the output.

---

## 17. Exam question to code table

| The question says | You write |
|---|---|
| Keep 25% as the test set, `random_state=0` | `train_test_split(X, y, test_size=0.25, random_state=0)` |
| Replace the unknown values with `np.nan` | `df.loc[df["RM"] < 1, "RM"] = np.nan` (and so on) |
| Impute with the mean, then min max scaling | a group with `make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler())` |
| Impute with the most frequent value, then one hot encode with `drop='first'` | a group with `make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(drop="first", sparse_output=False))` |
| Ordinal encode with this order | `OrdinalEncoder(categories=[[...]])` |
| Replace unseen test values with -1 | `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` |
| Keep the remaining columns as they are | `remainder="passthrough"` |
| How many features after preprocessing | `X_train_t.shape[1]` |
| Mean of the transformed training data (whole matrix) | `X_train_t.mean()` |
| Highest value in the transformed test matrix | `X_test_t.max()` |
| Number of samples in the test set | `X_test.shape[0]` |
| Select the K best features by f_regression | `SelectKBest(score_func=f_regression, k=K)` |
| Which features does RFE eliminate | `np.where(~rfe.support_)[0]` |
| Fraction explained by the first principal component | `pca.explained_variance_ratio_[0]` |
| How many components keep 95% | `PCA(n_components=0.95)`, then `pca.n_components_` |

---

## 18. Counting features after preprocessing

For every feature, count how many columns it contributes. Then add them up.

| Treatment | Columns produced |
|---|---|
| numeric column (impute, scale) | 1 |
| ordinal encoding | 1 |
| one-hot encoding | number of unique words in train |
| one-hot with `drop="first"` | unique words minus 1 |
| a column you do not mention | 0 (dropped), or 1 if `remainder="passthrough"` |
| a column you drop | 0 |

The original text columns are replaced, so you never count them in addition to the new ones.

Example: 7 columns, with `city` (6 unique cities, one-hot), `grade` (ordinal) and 5 numeric columns. Count: 6 + 1 + 5 = 12. With `drop="first"` on `city`: 5 + 1 + 5 = 11.

---

## 19. Reading errors

Always read the last line of the error first. It tells you what went wrong.

| Error | Means | Fix |
|---|---|---|
| `KeyError: "None of ['grade'] are in the columns"` | that column name is not in this table | check the table you are using and the spelling |
| `NameError: name 'X' is not defined` | Python has never seen this name | import it, create it, or fix the spelling |
| `AttributeError: ... has no attribute 'tranform'` | a misspelled method or attribute | check the spelling (`transform`) |
| `ValueError: could not convert string to float: 'F'` | a tool that needs numbers received text | encode the text first, or pass only numeric columns |
| `ValueError: 2` when assigning a transform | a 2D result assigned to a single-column slot | use double brackets on both sides of the `=` |
| `TypeError: ... got an unexpected keyword argument 'sparse_output'` | a setting that this tool does not have (for example on `OrdinalEncoder`) | check the setting with `help()` |
| `TypeError: 'Index' object is not callable` | brackets after something that is not a function | write `df.columns` without brackets |

---

## 20. Common pitfalls

| Pitfall | Correct approach |
|---|---|
| Fitting on the test data | fit on train only, transform both |
| Forgetting double brackets | `X_train[["income"]]` on both sides |
| Reusing one variable name for different tools | one name per tool: `imp_income`, `imp_policy` |
| Printing several things but seeing only the last | use `print()` for each |
| Running a scaling cell twice on an overwritten column | rebuild the column, or write to a new variable |
| Alphabetical ordinal order | always pass `categories=[[...]]` |
| Using `drop="first"` when the question did not ask | use it only when asked |
| Test size rounding | 30% of 12 is 3.6, rounded up to 4 |
| Using `fit_transform` on test | test gets `transform` only |
| Copying the `random_state` wrongly | copy it exactly |
| Counting the original text column as well as the new columns | the originals are replaced |
| `handle_unknown="ignore"` on `OrdinalEncoder` | ordinal uses `use_encoded_value` and `unknown_value=-1` |
| Passing text columns into PCA or a selector | encode first |
| Mean of a matrix with NaN left in | impute before scaling |

---

## 21. Practice exercises

1. A dataset has 20 rows. How many rows are in the test set with `test_size=0.3`? What about 15 rows and `test_size=0.2`?
2. Write the line that fills the missing `income` of `X_train` with the median learned from train.
3. A column `grade` holds Low, Medium, High. Write the encoder. Then explain why the default order would be wrong.
4. How many columns does a one-hot of a column with 5 unique words give? And with `drop="first"`?
5. Train ages have min 28 and max 60. What does a test age of 20 become with MinMaxScaler?
6. A table has 7 features: `city` (6 cities, one-hot), `grade` (ordinal) and 5 numeric columns. How many features after preprocessing? How many if `city` uses `drop="first"`?
7. Write a ColumnTransformer for: `age` min max scaling, `income` impute with the median then standard scaling, `gender` one hot encoding. Keep the other columns.
8. What is the difference between a ColumnTransformer and a Pipeline?
9. You use `SelectKBest(f_regression, k=4)` on 10 columns. How many columns come out? Does it keep the highest or the lowest scores?
10. RFE reports `ranking_ = [1 1 1 2 3]`. Which columns were kept, and which was eliminated first?
11. `explained_variance_ratio_` is `[0.60, 0.25, 0.10]`. What is the total? What is the first component's share?
12. A tool raises `ValueError: could not convert string to float: 'F'`. What is wrong?

---

## 22. End-term quick revision

Concepts that an end-term question may ask in words:

- Why split the data? To judge the model on data it has never seen.
- What is data leakage? Test information reaching the learning step. It makes the score look better than it is. Avoid it by fitting on train only.
- Why scale? Columns on large scales would dominate many models, and models trained by gradient steps need similar scales to converge well.
- StandardScaler versus MinMaxScaler: standard centres at mean 0 with standard deviation 1. MinMax squeezes the train data into 0 to 1 and is sensitive to extreme values.
- Why one-hot and not just numbers? Numbers imply an order and distance that the categories do not have.
- Ordinal versus one-hot: use ordinal when the words have a natural order.
- Why `drop="first"`? The dropped column can be worked out from the others, so keeping it adds repeated information.
- What is imputation? Filling missing values with a statistic learned from train.
- ColumnTransformer versus Pipeline: the first applies different tools to different columns side by side. The second chains tools on the same data in order.
- Filter versus wrapper selection: SelectKBest scores columns one at a time and is fast. RFE and sequential selection use a model, so they see columns together, and they are slower.
- Feature selection versus PCA: selection keeps some of the original columns. PCA builds new columns that mix the old ones.
- What does `explained_variance_ratio_` mean? The share of the total variance carried by each component.