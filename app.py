"""Ensemble Learning Dashboard - Sri Lankan used-vehicle prices.
Run:  streamlit run app.py     (after running train_models.py or the notebook once)"""
import json, joblib
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px
import streamlit as st
from hybrid import HybridRFANN  # noqa: F401  (needed to load the hybrid model)

st.set_page_config(page_title="Ensemble Learning Dashboard - Sri Lanka", layout="wide")


@st.cache_data
def load_tables():
    meta = json.load(open("results/meta.json"))
    return (meta, pd.read_csv("data/vehicles_clean.csv"), pd.read_csv("results/metrics.csv"),
            pd.read_csv("results/predictions.csv"), pd.read_csv("results/feature_importance.csv"))


@st.cache_resource
def load_model(fname):
    return joblib.load(Path("models") / fname)


if not Path("results/meta.json").exists():
    st.error("Run `python train_models.py` (or the notebook) first to create models/ and results/.")
    st.stop()

meta, df, metrics, preds, imp = load_tables()
pre = joblib.load("models/preprocessor.joblib")
best = meta["best_model"]

st.title("Sri Lankan Used-Vehicle Price Predictor")
st.caption("Ensemble Learning Dashboard: Bagging + Boosting + Stacking + Hybrid (RF + ANN)")
if "synthetic" not in meta["data_file"].lower():
    st.caption(f"Data source: {meta['data_file']} | {meta['n_rows']:,} cleaned listings")
if "synthetic" in meta["data_file"].lower():
    st.warning("This dashboard is using SYNTHETIC test data. Replace it with real Sri Lankan data before submitting.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset", "Model comparison", "Actual vs predicted", "Feature importance", "Predict a price"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Clean rows", f"{meta['n_rows']:,}"); c2.metric("Train / test", f"{meta['n_train']:,} / {meta['n_test']:,}")
    c3.metric("Median price (LKR)", f"{df['price_lkr'].median():,.0f}")
    st.dataframe(df.sample(min(10, len(df)), random_state=1), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="price_lkr", nbins=40, title="Price distribution (LKR)"), width="stretch")
    b.plotly_chart(px.scatter(df, x="age", y="price_lkr", color="fuel", opacity=.5, title="Price vs age"), width="stretch")

with tab2:
    fam = st.multiselect("Show model families", sorted(metrics["Family"].unique()), default=sorted(metrics["Family"].unique()))
    m = metrics[metrics["Family"].isin(fam)].copy()
    m["Best"] = np.where(m["Model"] == best, "BEST", "")
    st.dataframe(m.round(3), width="stretch")
    metric = st.selectbox("Metric", ["RMSE", "MAE", "R2", "MAPE_%", "Train_time_s"])
    fig = px.bar(m.sort_values(metric, ascending=(metric != "R2")), x=metric, y="Model", orientation="h",
                 color=np.where(m.sort_values(metric, ascending=(metric != "R2"))["Model"] == best, "Best", "Other"),
                 title=f"{metric} by model (best overall by RMSE: {best})")
    st.plotly_chart(fig, width="stretch")

with tab3:
    sel = st.selectbox("Model selector", list(meta["model_files"]), index=list(meta["model_files"]).index(best))
    d = pd.DataFrame({"actual": preds["actual_lkr"], "predicted": preds[sel]})
    fig = px.scatter(d, x="actual", y="predicted", opacity=.5, title=f"Actual vs predicted - {sel}")
    fig.add_shape(type="line", x0=0, y0=0, x1=d.actual.max(), y1=d.actual.max(), line=dict(dash="dash", color="red"))
    st.plotly_chart(fig, width="stretch")

with tab4:
    tree_models = sorted(imp["model"].unique())
    pick = st.selectbox("Model", tree_models)
    s = imp[imp["model"] == pick].sort_values("importance")
    st.plotly_chart(px.bar(s, x="importance", y="feature", orientation="h", title=f"Feature importance - {pick}"), width="stretch")

with tab5:
    st.subheader("Make a prediction")
    left, right = st.columns(2)
    inp = {}
    for i, col in enumerate(meta["cat_cols"]):
        inp[col] = (left if i % 2 == 0 else right).selectbox(col.title(), meta["categories"][col])
    lo, med, hi = meta["num_ranges"]["age"]; inp["age"] = left.slider("Vehicle age (years)", int(lo), int(hi), int(med))
    lo, med, hi = meta["num_ranges"]["mileage_km"]; inp["mileage_km"] = right.number_input("Mileage (km)", 0, int(hi * 2), int(med), step=1000)
    lo, med, hi = meta["num_ranges"]["engine_cc"]; inp["engine_cc"] = left.number_input("Engine (cc)", 0, int(hi * 2), int(med), step=50)
    asking = right.number_input("Seller's asking price in LKR (optional - for the Deal Score)", 0, 500_000_000, 0, step=100_000)
    which = st.selectbox("Model to use", list(meta["model_files"]), index=list(meta["model_files"]).index(best), key="pm")
    if st.button("Predict price"):
        x = pre.transform(pd.DataFrame([inp])[meta["cat_cols"] + meta["num_cols"]])
        out = {n: float(np.expm1(load_model(f).predict(x))[0]) for n, f in meta["model_files"].items()}
        st.success(f"{which}: estimated price Rs. {out[which]:,.0f}")
        if asking > 0:   # innovation: Deal Score = asking price vs model estimate
            gap = (asking - out[which]) / out[which] * 100
            verdict = "looks OVERPRICED" if gap > 10 else ("looks like a FAIR price" if gap >= -10 else "looks like a GOOD DEAL (check why it is cheap!)")
            st.info(f"Deal Score: asking price is {gap:+.1f}% vs the model estimate - {verdict}.")
        st.write("All models:")
        st.dataframe(pd.DataFrame({"Model": out.keys(), "Predicted price (LKR)": [f"{v:,.0f}" for v in out.values()]}), width="stretch")
