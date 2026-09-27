#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, catboost, lightgbm
print(sys.executable)


# In[2]:


import numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

DATA = Path(".")  # on Kaggle: Path("/kaggle/input/dsn-bootcamp-qualification-hackathon-2026-ml-track")
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

ID, TARGET = "id", "total_sales"
expected = ["product_code","product_weight_kg","fat_content","shelf_visibility","product_category",
            "product_price","store_code","store_age_years","store_size","store_location_tier","store_format"]
missing_cols = [c for c in expected if c not in test.columns]
print("shapes:", train.shape, test.shape, "| missing expected columns:", missing_cols)
print(train.dtypes)


# In[3]:


print(train[TARGET].describe(), "\nskew:", train[TARGET].skew())
print(train.isna().mean().round(3)[lambda s: s > 0])
for c in train.select_dtypes(exclude="number").columns.drop(ID, errors="ignore"):
    print(c, train[c].nunique(), list(train[c].unique()[:8]))
print(train.select_dtypes("number").corr()[TARGET].sort_values())
print(train.groupby("store_format")[TARGET].mean().sort_values())
print(train.groupby("store_location_tier")[TARGET].mean().sort_values())

# Is sales ~ price x whole units? If the second line is near 0, the "ratio" model below should help.
r = train[TARGET] / train["product_price"]
print(r.describe()); print((r - r.round()).abs().describe())


# In[4]:


print(X.shape, X_test.shape)
print("categories after cleaning:", feat["product_category"].nunique())
print(feat[NUM].isna().sum()[lambda s: s > 0])


# In[3]:


full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
n_tr = len(train)

def clean_and_engineer(df):
    df = df.copy()
    # text standardisation (fixes casing/label fragmentation)
    for c in ["product_category","fat_content","store_size","store_location_tier","store_format"]:
        df[c] = df[c].str.strip().str.lower()
    df["fat_content"] = df["fat_content"].replace({"lf": "low fat", "reg": "regular"})

    # weight: product mean -> category mean -> global mean
    g = df.groupby("product_code")
    df["product_weight_kg"] = df["product_weight_kg"].fillna(g["product_weight_kg"].transform("mean"))
    df["product_weight_kg"] = (df["product_weight_kg"]
        .fillna(df.groupby("product_category")["product_weight_kg"].transform("mean"))
        .fillna(df["product_weight_kg"].mean()))

    # visibility: exact zeros are impossible -> missing -> product mean -> category mean -> global mean
    vis = df["shelf_visibility"].replace(0, np.nan)
    df["shelf_visibility"] = (vis.fillna(vis.groupby(df["product_code"]).transform("mean"))
        .fillna(vis.groupby(df["product_category"]).transform("mean")).fillna(vis.mean()))

    # store size: mode within store, explicit "unknown" if a store never reports it
    mode_fill = df.groupby("store_code")["store_size"].transform(
        lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
    df["store_size"] = df["store_size"].fillna(mode_fill).fillna("unknown")

    # features that summarise a product / store across rows (no target used -> no leakage)
    g = df.groupby("product_code")
    df["product_avg_price"] = g["product_price"].transform("mean")
    df["product_num_stores"] = g["store_code"].transform("nunique")
    df["price_deviation"] = df["product_price"] - df["product_avg_price"]
    s = df.groupby("store_code")
    df["store_num_products"] = s["product_code"].transform("nunique")
    df["store_avg_price"] = s["product_price"].transform("mean")
    df["format_x_tier"] = df["store_format"] + "_" + df["store_location_tier"]
    return df

feat = clean_and_engineer(full)
CAT = ["product_code","product_category","fat_content","store_code","store_size",
       "store_location_tier","store_format","format_x_tier"]
NUM = [c for c in feat.columns if c not in CAT + [ID]]

X_all = feat[CAT + NUM].copy()
for c in CAT:
    X_all[c] = X_all[c].fillna("unknown").astype(str)
assert X_all[NUM].select_dtypes(exclude="number").empty, "unexpected non-numeric column in NUM"

X, X_test = X_all.iloc[:n_tr].reset_index(drop=True), X_all.iloc[n_tr:].reset_index(drop=True)

# LightGBM copy: drop the high-cardinality id, use category dtype
X_lgb_all = X_all.drop(columns=["product_code"]).copy()
for c in [c for c in CAT if c != "product_code"]:
    X_lgb_all[c] = X_lgb_all[c].astype("category")
X_lgb, X_lgb_test = X_lgb_all.iloc[:n_tr].reset_index(drop=True), X_lgb_all.iloc[n_tr:].reset_index(drop=True)

y = train[TARGET].values
price_tr, price_te = X["product_price"].values, X_test["product_price"].values
assert (price_tr > 0).all() and (price_te > 0).all()


# In[6]:


print(X.shape, X_test.shape)
print("categories after cleaning:", feat["product_category"].nunique())
print(feat[NUM].isna().sum()[lambda s: s > 0])


# In[4]:


from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

# CatBoost is happiest with plain object columns for its categorical features (pandas 3 uses a new string type)
for c in CAT:
    X[c] = X[c].astype(object)
    X_test[c] = X_test[c].astype(object)

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

# "ratio" = predict units (sales/price), weighted by price^2 so the loss matches raw-scale RMSE
TRANS = {
    "raw":   dict(f=lambda y, p: y,          inv=lambda z, p: z,         w=lambda p: None),
    "sqrt":  dict(f=lambda y, p: np.sqrt(y), inv=lambda z, p: z ** 2,    w=lambda p: None),
    "ratio": dict(f=lambda y, p: y / p,      inv=lambda z, p: z * p,     w=lambda p: p ** 2 / np.mean(p ** 2)),
}

def cv_run(name, fit_predict, Xtr, Xte, trans="raw", n_splits=5, seeds=(42,)):
    T = TRANS[trans]
    oof, pred = np.zeros(len(Xtr)), np.zeros(len(Xte))
    for seed in seeds:
        for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(Xtr):
            p_a, p_b = price_tr[tr], price_tr[va]
            pv, pt = fit_predict(Xtr.iloc[tr], T["f"](y[tr], p_a), T["w"](p_a),
                                 Xtr.iloc[va], T["f"](y[va], p_b), T["w"](p_b), Xte)
            oof[va] += T["inv"](pv, p_b) / len(seeds)
            pred += T["inv"](pt, price_te) / (n_splits * len(seeds))
    print(f"{name:24s} CV RMSE = {rmse(y, oof):8.2f}")
    return oof, pred

CB_PARAMS = dict(iterations=3000, learning_rate=0.03, depth=4, l2_leaf_reg=5, loss_function="RMSE",
                 boosting_type="Ordered", early_stopping_rounds=200, random_seed=42, verbose=0)
LGB_PARAMS = dict(n_estimators=5000, learning_rate=0.02, num_leaves=8, max_depth=4, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10,
                  random_state=42, verbose=-1)

def fit_cb(params):
    def _fit(Xa, ya, wa, Xb, yb, wb, Xt):
        cats = [c for c in CAT if c in Xa.columns]
        m = CatBoostRegressor(**params)
        m.fit(Pool(Xa, ya, weight=wa, cat_features=cats),
              eval_set=Pool(Xb, yb, weight=wb, cat_features=cats))
        return m.predict(Xb), m.predict(Xt)
    return _fit

def fit_lgb(params):
    def _fit(Xa, ya, wa, Xb, yb, wb, Xt):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xb, yb)],
              eval_sample_weight=None if wb is None else [wb],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        return m.predict(Xb), m.predict(Xt)
    return _fit


# In[8]:


import time
t0 = time.time()
runs = {}
runs["cb_raw"] = cv_run("CatBoost raw", fit_cb(CB_PARAMS), X, X_test, "raw")
print(f"took {time.time() - t0:.0f}s")


# In[9]:


sub = pd.DataFrame({ID: test[ID], TARGET: np.clip(runs["cb_raw"][1], 0, None)})
assert list(sub.columns) == list(sample.columns), (list(sub.columns), list(sample.columns))
sub.to_csv("submission_v1.csv", index=False)
print(sub.shape); print(sub.head())


# In[10]:


runs["cb_ratio"]  = cv_run("CatBoost ratio", fit_cb(CB_PARAMS),   X,     X_test,     "ratio")
runs["cb_sqrt"]   = cv_run("CatBoost sqrt",  fit_cb(CB_PARAMS),   X,     X_test,     "sqrt")
runs["lgb_raw"]   = cv_run("LightGBM raw",   fit_lgb(LGB_PARAMS), X_lgb, X_lgb_test, "raw")
runs["lgb_ratio"] = cv_run("LightGBM ratio", fit_lgb(LGB_PARAMS), X_lgb, X_lgb_test, "ratio")


# In[11]:


from scipy.optimize import nnls

keys = list(runs)
O = np.column_stack([runs[k][0] for k in keys])
P = np.column_stack([runs[k][1] for k in keys])
w, _ = nnls(O, y); w /= w.sum()
print(dict(zip(keys, w.round(3))), "| blend CV RMSE:", round(rmse(y, O @ w), 2))

final = np.clip(P @ w, 0, None)
sub = pd.DataFrame({ID: test[ID], TARGET: final})
assert list(sub.columns) == list(sample.columns)
sub.to_csv("submission_v2_blend.csv", index=False)


# In[12]:


import pickle
pickle.dump(runs, open("runs.pkl", "wb"))


# In[ ]:


print(X.shape, list(runs))


# In[1]:


print("ok")


# In[5]:


import pickle
runs = pickle.load(open("runs.pk1", "rb"))
print(X.shape, list(runs))


# In[6]:


import pickle
runs = pickle.load(open("runs.pkl", "rb"))
print(X.shape, list(runs))


# In[7]:


for combo in [["cb_ratio","cb_raw"], ["cb_ratio","cb_sqrt"], ["cb_ratio","cb_raw","cb_sqrt","lgb_ratio"]]:
    print(combo, round(rmse(y, np.mean([runs[k][0] for k in combo], axis=0)), 2))


# In[8]:


runs["cb_ratio_3s"] = cv_run("CatBoost ratio 3 seeds", fit_cb(CB_PARAMS), X, X_test, "ratio", seeds=(42, 7, 2024))

sub = pd.DataFrame({ID: test[ID], TARGET: np.clip(runs["cb_ratio_3s"][1], 0, None)})
assert list(sub.columns) == list(sample.columns)
sub.to_csv("submission_v2_cb_ratio_3seeds.csv", index=False)

pickle.dump(runs, open("runs.pkl", "wb"))


# In[9]:


chk = pd.read_csv("submission_v2_cb_ratio_3seeds.csv")
print(chk.shape, list(chk.columns))
print("nulls:", chk.isna().sum().sum(), "| negatives:", (chk[TARGET] < 0).sum())
print("ids match test:", (chk[ID].values == test[ID].values).all())


# In[1]:


for d, l2 in [(3, 5), (5, 5), (4, 10), (4, 2)]:
    p = {**CB_PARAMS, "depth": d, "l2_leaf_reg": l2}
    runs[f"cb_ratio_d{d}_l{l2}"] = cv_run(f"CB ratio d={d} l2={l2}", fit_cb(p), X, X_test, "ratio")

pickle.dump(runs, open("runs.pkl", "wb"))


# In[2]:


from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

TRANS = {
    "raw":   dict(f=lambda y, p: y,          inv=lambda z, p: z,         w=lambda p: None),
    "sqrt":  dict(f=lambda y, p: np.sqrt(y), inv=lambda z, p: z ** 2,    w=lambda p: None),
    "ratio": dict(f=lambda y, p: y / p,      inv=lambda z, p: z * p,     w=lambda p: p ** 2 / np.mean(p ** 2)),
}

def cv_run(name, fit_predict, Xtr, Xte, trans="raw", n_splits=5, seeds=(42,)):
    T = TRANS[trans]
    oof, pred = np.zeros(len(Xtr)), np.zeros(len(Xte))
    for seed in seeds:
        for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(Xtr):
            p_a, p_b = price_tr[tr], price_tr[va]
            pv, pt = fit_predict(Xtr.iloc[tr], T["f"](y[tr], p_a), T["w"](p_a),
                                 Xtr.iloc[va], T["f"](y[va], p_b), T["w"](p_b), Xte)
            oof[va] += T["inv"](pv, p_b) / len(seeds)
            pred += T["inv"](pt, price_te) / (n_splits * len(seeds))
    print(f"{name:24s} CV RMSE = {rmse(y, oof):8.2f}")
    return oof, pred

CB_PARAMS = dict(iterations=3000, learning_rate=0.03, depth=4, l2_leaf_reg=5, loss_function="RMSE",
                 boosting_type="Ordered", early_stopping_rounds=200, random_seed=42, verbose=0)
LGB_PARAMS = dict(n_estimators=5000, learning_rate=0.02, num_leaves=8, max_depth=4, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10,
                  random_state=42, verbose=-1)

def fit_cb(params):
    def _fit(Xa, ya, wa, Xb, yb, wb, Xt):
        cats = [c for c in CAT if c in Xa.columns]
        m = CatBoostRegressor(**params)
        m.fit(Pool(Xa, ya, weight=wa, cat_features=cats),
              eval_set=Pool(Xb, yb, weight=wb, cat_features=cats))
        return m.predict(Xb), m.predict(Xt)
    return _fit

def fit_lgb(params):
    def _fit(Xa, ya, wa, Xb, yb, wb, Xt):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xb, yb)],
              eval_sample_weight=None if wb is None else [wb],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        return m.predict(Xb), m.predict(Xt)
    return _fit


# In[3]:


for d, l2 in [(3, 5), (5, 5), (4, 10), (4, 2)]:
    p = {**CB_PARAMS, "depth": d, "l2_leaf_reg": l2}
    runs[f"cb_ratio_d{d}_l{l2}"] = cv_run(f"CB ratio d={d} l2={l2}", fit_cb(p), X, X_test, "ratio")

pickle.dump(runs, open("runs.pkl", "wb"))


# In[4]:


import numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

DATA = Path(".")
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")
ID, TARGET = "id", "total_sales"

full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
n_tr = len(train)

def clean_and_engineer(df):
    df = df.copy()
    for c in ["product_category","fat_content","store_size","store_location_tier","store_format"]:
        df[c] = df[c].str.strip().str.lower()
    df["fat_content"] = df["fat_content"].replace({"lf": "low fat", "reg": "regular"})

    g = df.groupby("product_code")
    df["product_weight_kg"] = df["product_weight_kg"].fillna(g["product_weight_kg"].transform("mean"))
    df["product_weight_kg"] = (df["product_weight_kg"]
        .fillna(df.groupby("product_category")["product_weight_kg"].transform("mean"))
        .fillna(df["product_weight_kg"].mean()))

    vis = df["shelf_visibility"].replace(0, np.nan)
    df["shelf_visibility"] = (vis.fillna(vis.groupby(df["product_code"]).transform("mean"))
        .fillna(vis.groupby(df["product_category"]).transform("mean")).fillna(vis.mean()))

    mode_fill = df.groupby("store_code")["store_size"].transform(
        lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
    df["store_size"] = df["store_size"].fillna(mode_fill).fillna("unknown")

    g = df.groupby("product_code")
    df["product_avg_price"] = g["product_price"].transform("mean")
    df["product_num_stores"] = g["store_code"].transform("nunique")
    df["price_deviation"] = df["product_price"] - df["product_avg_price"]
    s = df.groupby("store_code")
    df["store_num_products"] = s["product_code"].transform("nunique")
    df["store_avg_price"] = s["product_price"].transform("mean")
    df["format_x_tier"] = df["store_format"] + "_" + df["store_location_tier"]
    return df

feat = clean_and_engineer(full)
CAT = ["product_code","product_category","fat_content","store_code","store_size",
       "store_location_tier","store_format","format_x_tier"]
NUM = [c for c in feat.columns if c not in CAT + [ID]]

X_all = feat[CAT + NUM].copy()
for c in CAT:
    X_all[c] = X_all[c].fillna("unknown").astype(object)

X, X_test = X_all.iloc[:n_tr].reset_index(drop=True), X_all.iloc[n_tr:].reset_index(drop=True)

X_lgb_all = X_all.drop(columns=["product_code"]).copy()
for c in [c for c in CAT if c != "product_code"]:
    X_lgb_all[c] = X_lgb_all[c].astype("category")
X_lgb, X_lgb_test = X_lgb_all.iloc[:n_tr].reset_index(drop=True), X_lgb_all.iloc[n_tr:].reset_index(drop=True)

y = train[TARGET].values
price_tr, price_te = X["product_price"].values, X_test["product_price"].values

print("ready:", X.shape, X_test.shape)


# In[5]:


for d, l2 in [(3, 5), (5, 5), (4, 10), (4, 2)]:
    p = {**CB_PARAMS, "depth": d, "l2_leaf_reg": l2}
    runs[f"cb_ratio_d{d}_l{l2}"] = cv_run(f"CB ratio d={d} l2={l2}", fit_cb(p), X, X_test, "ratio")


# In[6]:


runs = {}
runs["cb_ratio_d3_l5"] = (None, None)  # placeholder, will be overwritten below

for d, l2 in [(3, 5), (5, 5), (4, 10), (4, 2)]:
    p = {**CB_PARAMS, "depth": d, "l2_leaf_reg": l2}
    runs[f"cb_ratio_d{d}_l{l2}"] = cv_run(f"CB ratio d={d} l2={l2}", fit_cb(p), X, X_test, "ratio")


# In[7]:


drop_cols = ["product_weight_kg", "store_age_years"]
X_trim, X_test_trim = X.drop(columns=drop_cols), X_test.drop(columns=drop_cols)
runs["cb_ratio_trimmed"] = cv_run("CB ratio trimmed", fit_cb(CB_PARAMS), X_trim, X_test_trim, "ratio")


# In[8]:


keys = list(runs)
O = np.column_stack([runs[k][0] for k in keys])
P = np.column_stack([runs[k][1] for k in keys])
avg_oof, avg_pred = O.mean(axis=1), P.mean(axis=1)
print("simple average of all runs, CV RMSE:", round(rmse(y, avg_oof), 2))


# In[9]:


from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

# one-hot the categoricals, keep price and its interactions with store/format
ohe_cols = ["store_code", "store_format", "store_location_tier", "fat_content"]
ohe = OneHotEncoder(handle_unknown="ignore")
Xc = ohe.fit_transform(X[ohe_cols].astype(str))
Xtc = ohe.transform(X_test[ohe_cols].astype(str))

price_col = X[["product_price"]].values
price_col_t = X_test[["product_price"]].values

# interact price with each store dummy (lets slope vary by store)
Xlin = sparse.hstack([Xc, Xc.multiply(price_col), price_col]).tocsr()
Xlin_t = sparse.hstack([Xtc, Xtc.multiply(price_col_t), price_col_t]).tocsr()

def fit_ridge(Xa, ya, wa, Xb, yb, wb, Xt):
    m = Ridge(alpha=5.0)
    m.fit(Xa, ya, sample_weight=wa)
    return m.predict(Xb), m.predict(Xt)

# reindex sparse rows the same way cv_run expects (needs .iloc, so wrap as array-index-friendly)
class SparseWrap:
    def __init__(self, M): self.M = M
    def iloc(self): pass

# cv_run expects .iloc[idx] - do a lightweight manual CV instead:
oof, pred = np.zeros(len(X)), np.zeros(len(X_test))
for tr, va in KFold(5, shuffle=True, random_state=42).split(X):
    pa, pb = price_tr[tr], price_tr[va]
    ya, yb = y[tr] / pa, y[va] / pb
    wa = pa ** 2 / np.mean(pa ** 2)
    m = Ridge(alpha=5.0)
    m.fit(Xlin[tr], ya, sample_weight=wa)
    oof[va] = m.predict(Xlin[va]) * pb
    pred += m.predict(Xlin_t) * price_te / 5

runs["ridge_ratio"] = (oof, pred)
print("Ridge (linear) ratio CV RMSE:", round(rmse(y, oof), 2))


# In[10]:


w_grid = [0.0, 0.1, 0.2, 0.3]
for w in w_grid:
    blend = (1 - w) * runs["cb_ratio_3s"][0] + w * runs["ridge_ratio"][0]
    print(f"ridge weight {w}: CV RMSE = {rmse(y, blend):.2f}")


# In[11]:


w_grid = [0.0, 0.1, 0.2, 0.3]
for w in w_grid:
    blend = (1 - w) * runs["cb_ratio_d4_l2"][0] + w * runs["ridge_ratio"][0]
    print(f"ridge weight {w}: CV RMSE = {rmse(y, blend):.2f}")


# In[ ]:




