# %% [markdown]
# # Sri Lankan Used-Vehicle Price Predictor - Ensemble Learning Dashboard
# **Author:** [YOUR NAME / INDEX NO.]  |  **Module:** [MODULE CODE]  |  **Date:** 23 Nov 2026
#
# **AI-use disclosure (required by lecturer):** The code skeleton and dashboard front end in this notebook were
# generated with the help of Claude (Anthropic). I ran it on my own data, checked the results,
# and wrote the interpretation sections myself. [EDIT THIS SENTENCE TO MATCH WHAT YOU REALLY DID]
#
# **Problem & innovation:** [WRITE 3-4 SENTENCES: who needs fair used-car price estimates in Sri Lanka and what is new in your idea]

# %% [markdown]
# ## 1. Setup

# %%
import sys, time, json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from hybrid import HybridRFANN

warnings.filterwarnings("ignore")
SEED = 42
TUNE = True                                   # set False for a faster run
DATA_PATH = "data/car_price_dataset.csv"
if not Path(DATA_PATH).exists():              # fall back to the synthetic TEST file
    DATA_PATH = "data/vehicles_SAMPLE_synthetic.csv"
    print("WARNING: using SYNTHETIC test data. Do NOT submit these results.")
for d in ["models", "results"]:
    Path(d).mkdir(exist_ok=True)
print("Data file:", DATA_PATH)

# %% [markdown]
# ## 2. Data source and cleaning
# Source: [riyasewana.com / ikman.lk / own collection - write exactly where, and when you collected it].
# Price text such as "Rs. 4,500,000" and mileage such as "85,000 km" are converted to numbers.

# %%
def to_number(s):
    """'Rs. 4,500,000' -> 4500000.0 ; '85,000 km' -> 85000.0 ; missing -> NaN"""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(s))          # first number in the text (ignores 'Rs.' and 'km')
    return float(m.group().replace(",", "")) if m else np.nan

def clean_vehicle_df(df):
    df = df.copy()
    if "Brand" in df.columns:
        df = df.rename(columns={
            "Brand": "make",
            "Model": "model",
            "YOM": "year",
            "Engine (cc)": "engine_cc",
            "Gear": "transmission",
            "Fuel Type": "fuel",
            "Millage(KM)": "mileage_km",
            "Town": "district",
            "Price": "price",
        })
        df["price"] = pd.to_numeric(df["price"], errors="coerce") * 100_000
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")
        df["engine_cc"] = pd.to_numeric(df["engine_cc"], errors="coerce")
        df = df.rename(columns={"price": "price_lkr"})
    df.columns = [c.strip().lower() for c in df.columns]
    if "price" in df.columns:
        for col in ["price", "mileage", "engine", "year"]:
            if col in df.columns:
                df[col] = df[col].map(to_number)
        df = df.rename(columns={"price": "price_lkr", "mileage": "mileage_km", "engine": "engine_cc"})
    for col in ["make", "model", "fuel", "transmission", "district"]:
        df[col] = df[col].astype(str).str.strip().str.title().replace({"Nan": np.nan})
    df = df.drop_duplicates()
    df = df.dropna(subset=["price_lkr", "year"])                       # target + year must exist
    df = df[df["year"].between(1990, 2026)]
    lo, hi = df["price_lkr"].quantile([0.01, 0.99])                    # remove extreme prices
    df = df[df["price_lkr"].between(lo, hi)]
    df["mileage_km"] = df["mileage_km"].where(df["mileage_km"] < 500_000)
    df["age"] = 2026 - df["year"]
    return df.reset_index(drop=True)

raw = pd.read_csv(DATA_PATH)
df = clean_vehicle_df(raw)
print(f"raw rows: {len(raw)} | clean rows: {len(df)}")
print("missing values:\n", df.isna().sum()[df.isna().sum() > 0])
df.head()

# %% [markdown]
# ## 3. EDA
# Each plot is followed by one line: what it tells us. [WRITE YOUR OWN LINES AFTER RUNNING]

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(df["price_lkr"] / 1e6, bins=40, color="#2a9d8f"); ax[0].set_title("Price distribution (million LKR)")
ax[1].scatter(df["age"], df["price_lkr"] / 1e6, s=6, alpha=.4, color="#e76f51"); ax[1].set_title("Price vs vehicle age")
df.groupby("make")["price_lkr"].median().sort_values().div(1e6).plot.barh(ax=ax[2], color="#264653")
ax[2].set_title("Median price by make (million LKR)")
plt.tight_layout(); plt.savefig("results/eda.png", dpi=120); plt.show()

# %% [markdown]
# ## 4. Train/test split and preprocessing
# ONE split and ONE preprocessing for every model, so the comparison is fair.
# The target is log(price) because prices are right-skewed. Metrics are reported back in LKR.

# %%
CAT_COLS = ["make", "model", "fuel", "transmission", "district"]
NUM_COLS = ["age", "mileage_km", "engine_cc"]
X, y_lkr = df[CAT_COLS + NUM_COLS], df["price_lkr"]
y = np.log1p(y_lkr)
X_tr, X_te, y_tr, y_te, ylkr_tr, ylkr_te = train_test_split(X, y, y_lkr, test_size=0.2, random_state=SEED)

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
pre = ColumnTransformer([
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False)), CAT_COLS),
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM_COLS),
])
Xtr = pre.fit_transform(X_tr); Xte = pre.transform(X_te)
feat_names = list(pre.get_feature_names_out())
print("train:", Xtr.shape, "test:", Xte.shape)

# %% [markdown]
# ## 5. Light hyper-parameter tuning (cross-validation)
# RandomizedSearchCV with 3-fold CV on the TRAINING set only, for Random Forest and XGBoost.

# %%
rf_params, xgb_params = {}, {}
if TUNE:
    rs = RandomizedSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [None, 12, 20], "min_samples_leaf": [1, 2, 4], "max_features": [0.5, 1.0]},
        n_iter=6, cv=3, scoring="neg_root_mean_squared_error", random_state=SEED).fit(Xtr, y_tr)
    rf_params = rs.best_params_
    rs = RandomizedSearchCV(XGBRegressor(random_state=SEED, n_jobs=-1),
        {"n_estimators": [300, 600], "max_depth": [3, 5, 7], "learning_rate": [0.03, 0.06, 0.1], "subsample": [0.8, 1.0]},
        n_iter=6, cv=3, scoring="neg_root_mean_squared_error", random_state=SEED).fit(Xtr, y_tr)
    xgb_params = rs.best_params_
print("RF best:", rf_params); print("XGB best:", xgb_params)

# %% [markdown]
# ## 6. Models: baseline, bagging, boosting, stacking, hybrids

# %%
def new_rf():  return RandomForestRegressor(**{"n_estimators": 300, **rf_params}, random_state=SEED, n_jobs=-1)
def new_xgb(): return XGBRegressor(**{"n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, **xgb_params}, random_state=SEED, n_jobs=-1)
def new_lgb(): return LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)
def new_cat(): return CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=SEED, verbose=0, allow_writing_files=False)

models = {
    "Linear Regression (baseline)": ("Baseline", LinearRegression()),
    "Random Forest":                ("Bagging",  new_rf()),
    "Extra Trees":                  ("Bagging",  ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
    "Bagging (decision trees)":     ("Bagging",  BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, random_state=SEED, n_jobs=-1)),
    "XGBoost":                      ("Boosting", new_xgb()),
    "LightGBM":                     ("Boosting", new_lgb()),
    "CatBoost":                     ("Boosting", new_cat()),
    "Stacking (RF+XGB+LGBM+Cat -> Ridge)": ("Stacking", StackingRegressor(
        estimators=[("rf", new_rf()), ("xgb", new_xgb()), ("lgb", new_lgb()), ("cat", new_cat())],
        final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1)),
    "Hybrid A: RF -> ANN (RF output as ANN input)": ("Hybrid", HybridRFANN(random_state=SEED)),
    "Hybrid B: RF + ANN blend": ("Hybrid", VotingRegressor([
        ("rf", new_rf()),
        ("ann", MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=600, random_state=SEED))])),
    "ANN (MLP) alone": ("ANN", MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=600, random_state=SEED)),
}

# %% [markdown]
# ## 7. Train and evaluate (same split, same metrics, training time recorded)

# %%
def metrics_lkr(y_true_lkr, pred_log):
    p = np.expm1(pred_log)
    return {"MAE": mean_absolute_error(y_true_lkr, p),
            "RMSE": mean_squared_error(y_true_lkr, p) ** 0.5,
            "R2": r2_score(y_true_lkr, p),
            "MAPE_%": mean_absolute_percentage_error(y_true_lkr, p) * 100}

rows, preds = [], {"actual_lkr": ylkr_te.values}
for name, (family, mdl) in models.items():
    t0 = time.time(); mdl.fit(Xtr, y_tr); fit_s = time.time() - t0
    pred = mdl.predict(Xte); preds[name] = np.expm1(pred)
    cv_r2 = np.nan
    if family in ("Bagging", "Boosting") and name != "Bagging (decision trees)":   # 3-fold CV R2 on log price
        cv_r2 = cross_val_score(mdl, Xtr, y_tr, cv=3, scoring="r2").mean()
    rows.append({"Model": name, "Family": family, **metrics_lkr(ylkr_te, pred), "CV_R2_log": cv_r2, "Train_time_s": fit_s})
    print(f"done: {name:48s} RMSE={rows[-1]['RMSE']:,.0f}  R2={rows[-1]['R2']:.3f}  ({fit_s:.1f}s)")
res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
best = res.loc[0, "Model"]
res.round(3)

# %% [markdown]
# ## 8. ONE comparison table + bar chart

# %%
print(res.round(3).to_string(index=False))
print("\nBest model (lowest RMSE):", best)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
colors = ["#e63946" if m == best else "#457b9d" for m in res["Model"]]
ax[0].barh(res["Model"][::-1], res["RMSE"][::-1] / 1e3, color=colors[::-1]); ax[0].set_title("RMSE (thousand LKR) - lower is better")
ax[1].barh(res["Model"][::-1], res["R2"][::-1], color=colors[::-1]); ax[1].set_title("R2 - higher is better")
plt.tight_layout(); plt.savefig("results/model_comparison.png", dpi=120); plt.show()

# %% [markdown]
# ## 9. Actual vs predicted (best model)

# %%
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(preds["actual_lkr"] / 1e6, preds[best] / 1e6, s=8, alpha=.5)
m = preds["actual_lkr"].max() / 1e6; ax.plot([0, m], [0, m], "r--")
ax.set_xlabel("Actual (million LKR)"); ax.set_ylabel("Predicted (million LKR)"); ax.set_title(f"Actual vs predicted - {best}")
plt.tight_layout(); plt.savefig("results/actual_vs_pred.png", dpi=120); plt.show()

# %% [markdown]
# ## 10. Explainability - feature importance
# One-hot columns are added back together so we can talk about "make", "age", "fuel" ...

# %%
def aggregate(importances):
    agg = {c: 0.0 for c in CAT_COLS + NUM_COLS}
    for n, v in zip(feat_names, importances):
        col = n.split("__", 1)[1]
        key = next((c for c in CAT_COLS if col.startswith(c + "_")), col)
        agg[key] += float(v)
    s = sum(agg.values()) or 1
    return {k: v / s for k, v in agg.items()}

imp_rows = []
for nm in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
    imp = models[nm][1].feature_importances_
    for k, v in aggregate(imp).items():
        imp_rows.append({"model": nm, "feature": k, "importance": v})
imp_df = pd.DataFrame(imp_rows)
imp_df.pivot(index="feature", columns="model", values="importance").sort_values("Random Forest").plot.barh(figsize=(8, 5))
plt.title("Feature importance (share of total)"); plt.tight_layout(); plt.savefig("results/feature_importance.png", dpi=120); plt.show()

try:   # optional SHAP plot (can be slow on big data)
    import shap
    sv = shap.TreeExplainer(models["XGBoost"][1]).shap_values(Xte[:300])
    shap.summary_plot(sv, Xte[:300], feature_names=feat_names, show=False, max_display=12)
    plt.tight_layout(); plt.savefig("results/shap_xgboost.png", dpi=120); plt.show()
except Exception as e:
    print("SHAP skipped:", e)

# %% [markdown]
# ## 11. Save everything for the dashboard

# %%
for name, (_, mdl) in models.items():
    joblib.dump(mdl, "models/" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + ".joblib")
joblib.dump(pre, "models/preprocessor.joblib")
res.to_csv("results/metrics.csv", index=False)
pd.DataFrame(preds).to_csv("results/predictions.csv", index=False)
imp_df.to_csv("results/feature_importance.csv", index=False)
df.to_csv("data/vehicles_clean.csv", index=False)
meta = {"data_file": DATA_PATH, "n_rows": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "best_model": best, "cat_cols": CAT_COLS, "num_cols": NUM_COLS,
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_COLS},
        "model_files": {n: re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_") + ".joblib" for n in models},
        "families": {n: f for n, (f, _) in models.items()},
        "num_ranges": {c: [float(df[c].min()), float(df[c].median()), float(df[c].max())] for c in NUM_COLS},
        "rf_params": rf_params, "xgb_params": {k: (float(v) if isinstance(v, float) else v) for k, v in xgb_params.items()}}
json.dump(meta, open("results/meta.json", "w"), indent=2, default=str)
print("Saved models/, results/. Next: streamlit run app.py")

# %% [markdown]
# ## 12. Research discussion (WRITE IN YOUR OWN WORDS using YOUR results)
# **RF vs gradient boosting.** Bagging builds many independent trees in parallel and averages them (lowers variance).
# Boosting builds trees one after another, each fixing the previous errors (lowers bias).
# Compare speed, tuning effort, over-fitting risk and categorical handling (CatBoost has native categorical support; LightGBM is fastest).
#
# **Which model won on my data and why?** [Use the table above: name the winner, the gap in RMSE/MAE, and explain - e.g. strong
# non-linear age x mileage effects, many one-hot columns, small data size ...]
#
# **Did the hybrid and the stacking help?** [Compare each against its parts. If the gain is tiny, say so honestly.]
#
# **Tabular vs image/text.** Tree ensembles are usually very strong on tabular data (mixed types, little tuning, handle missing data)
# but are weak on raw images/text where neural networks learn features automatically. [Add 2-3 points in your own words.]
#
# **Limitations.** [Asking prices, not final sale prices; data from one website and one date; scraped sample size; no condition/photos...]
#
# **References.** [Chen & Guestrin 2016 XGBoost; Ke et al. 2017 LightGBM; Prokhorenkova et al. 2018 CatBoost; Breiman 2001 Random Forests; add yours]
