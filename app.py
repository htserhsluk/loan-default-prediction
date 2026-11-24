"""Run from the project directory with streamlit run app.py."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan default explorer", layout="wide")
st.title("Loan default explorer")
st.caption("Historical LendingClub loans · 36-month term · classroom analysis")
meta_path = Path("artifacts/selected.json")
if not meta_path.exists():
    st.info("Prepare the sample and run python benchmark.py to enable scoring.")
    st.stop()
meta = json.loads(meta_path.read_text())
model_path = meta_path.parent / (meta["model"] + ".joblib")
if not model_path.exists():
    st.error("The selected pipeline is missing. Run python benchmark.py.")
    st.stop()

@st.cache_resource
def load_pipeline(path, modified):
    return joblib.load(path)

score_tab, results_tab = st.tabs(["Score a loan", "Benchmark results"])
with score_tab:
    st.write("Adjust the example application. Additional fields start at training medians or the most frequent category.")
    values = dict(meta["defaults"])
    with st.form("application"):
        cols = st.columns(3)
        for col, (key, label) in zip(cols, [("loan_amnt", "Requested loan ($)"), ("annual_inc", "Annual income ($)"),
                                             ("dti", "Debt-to-income ratio (%)")]):
            values[key] = col.number_input(label, min_value=0., value=values[key])
        with st.expander("Additional application and credit history"):
            columns = st.columns(3)
            for i, key in enumerate(k for k in values if k not in {"loan_amnt", "annual_inc", "dti"}):
                column = columns[i % 3]
                if key in meta["categories"]:
                    choices = meta["categories"][key]
                    values[key] = column.selectbox(key, choices, index=choices.index(values[key]))
                else:
                    values[key] = column.number_input(key, value=values[key])
            missing_fields = st.multiselect("Fields with no recorded value",
                                            [key for key in values if key not in {"annual_inc", "loan_amnt"}],
                                            help="Selected fields use the fitted pipeline's missing-value handling.")
        submitted = st.form_submit_button("Score application")
    if submitted:
        if values["loan_amnt"] <= 0 or values["annual_inc"] <= 0:
            st.error("Enter a positive loan amount and annual income.")
        else:
            for key in missing_fields:
                values[key] = float("nan")
            pipeline = load_pipeline(model_path, model_path.stat().st_mtime_ns)
            score = float(pipeline.predict_proba(pd.DataFrame([values]))[0, 1])
            st.metric("Default score", f"{score:.1%}")
            threshold = meta["threshold"]
            label = "Above" if score >= threshold else "Below"
            st.write(f"{label} the {threshold:.1%} validation threshold for {meta['model']}.")
with results_tab:
    st.subheader("Held-out results")
    results = pd.read_csv("reports/benchmark.csv").query("split == 'test'")
    st.dataframe(results.drop(columns="split"), hide_index=True)
    for figure in ["cohort", "precision-recall", "efficiency", "errors"]:
        path = Path("reports", figure + ".png")
        if path.exists():
            st.image(str(path))
