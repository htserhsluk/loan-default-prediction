"""
Loan Default Risk - interactive UI
==================================
Run from the project root:
    streamlit run app/streamlit_app.py

Pages
  * Overview                - project summary and headline numbers
  * Score an applicant      - live PD from every model, decision under a chosen
                              cost ratio, per-applicant explanation, what-if curve
  * Batch scoring           - upload a CSV of applicants, download scored file
  * Model benchmark         - leaderboard, interactive ROC / PR curves,
                              confusion matrix at any threshold
  * Lending policy simulator- cost ratio -> optimal threshold -> $ impact
  * Analyses                - Feature ablation (A), Efficiency (C), Explainability (F)
  * Data & EDA              - dataset, sampling, leakage, missing values
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (FIG_DIR, MODELS_DIR, OUTCOME_FILE, REPORTS_DIR,  # noqa: E402
                        SAMPLE_FILE, TABLE_DIR)
from src.features import (REQUIRED_APPLICANT_COLS, load_splits, onehot_source,  # noqa: E402
                          prepare_applicants)
from src.models import MODEL_ORDER  # noqa: E402
from src.plotting import GRID, INK_2, MODEL_COLORS, MUTED, SLOTS  # noqa: E402
from src.train_benchmark import slug  # noqa: E402

st.set_page_config(page_title="Loan Default Risk", page_icon="📉", layout="wide")

STATUS = {  # risk bands: icon + label + reserved status colour (never colour alone)
    "Low": ("🟢", "#0ca30c"), "Moderate": ("🟡", "#fab219"),
    "High": ("🟠", "#ec835a"), "Very high": ("🔴", "#d03b3b"),
}
FRIENDLY = {
    "int_rate": "Interest rate", "sub_grade_num": "LC sub-grade", "grade_num": "LC grade",
    "term_months": "Loan term", "fico_score": "FICO score", "dti": "Debt-to-income",
    "annual_inc": "Annual income", "loan_amnt": "Loan amount", "installment": "Monthly instalment",
    "loan_to_income": "Loan / income", "installment_to_income": "Instalment / income",
    "revol_util": "Revolving utilisation", "revol_bal": "Revolving balance",
    "credit_history_months": "Credit history length", "inq_last_6mths": "Inquiries (6 mths)",
    "home_ownership": "Home ownership", "purpose": "Loan purpose", "addr_state": "State",
    "verification_status": "Income verification", "emp_length_num": "Employment length",
    "emp_title_missing": "No employment title", "acc_open_past_24mths": "Accounts opened (24 mths)",
    "mort_acc": "Mortgage accounts", "delinq_2yrs": "Delinquencies (2 yrs)",
    "open_acc": "Open accounts", "total_acc": "Total accounts", "pub_rec": "Public records",
    "bc_open_to_buy": "Bankcard open-to-buy", "tot_hi_cred_lim": "Total credit limit",
    "avg_cur_bal": "Average current balance", "num_tl_op_past_12m": "Accounts opened (12 mths)",
    "mths_since_recent_inq": "Months since last inquiry", "tot_cur_bal": "Total current balance",
    "bc_util": "Bankcard utilisation", "total_bc_limit": "Bankcard limit",
    "mo_sin_rcnt_tl": "Months since newest account", "percent_bc_gt_75": "% bankcards > 75% used",
    "revol_bal_to_income": "Revolving balance / income", "mths_since_recent_bc": "Months since newest bankcard",
    "mo_sin_old_rev_tl_op": "Age of oldest revolving account", "mo_sin_old_il_acct": "Age of oldest instalment loan",
    "num_actv_rev_tl": "Active revolving accounts", "total_rev_hi_lim": "Total revolving limit",
    "total_bal_ex_mort": "Balance excl. mortgage", "total_il_high_credit_limit": "Instalment credit limit",
    "initial_list_status": "Listing status", "application_type": "Application type",
}


def nice(f):
    return FRIENDLY.get(f, f.replace("_", " "))


# ==========================================================================
# Cached loaders
# ==========================================================================
@st.cache_resource(show_spinner="Loading models ...")
def load_models():
    models = {}
    for n in MODEL_ORDER:
        p = MODELS_DIR / f"{slug(n)}.joblib"
        if p.exists():
            models[n] = joblib.load(p)
    p = MODELS_DIR / "calibrators.joblib"
    cals = joblib.load(p) if p.exists() else {}   # isotonic calibrator per model
    return models, cals


@st.cache_data(show_spinner="Loading data ...")
def load_reference():
    d = load_splits()
    spec = d["spec"]
    cols = spec["numeric"] + spec["categorical"]
    tr = d["X_train"]
    template = {c: (float(tr[c].median()) if c in spec["numeric"] else str(tr[c].mode().iloc[0]))
                for c in cols}
    rate_by_sub = d["frame"].groupby("sub_grade_num")["int_rate"].median().to_dict()
    choices = {c: sorted(tr[c].dropna().astype(str).unique().tolist()) for c in spec["categorical"]}
    q = {c: tr[c].quantile([0.01, 0.99]).tolist() for c in spec["numeric"]}
    return template, rate_by_sub, choices, q, spec


@st.cache_data
def load_table(name):
    p = TABLE_DIR / name
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return json.loads(p.read_text())


def has(name):
    return (TABLE_DIR / name).exists()


def fig_path(sub, name):
    return FIG_DIR / sub / f"{name}.png"


def show_fig(sub, name, caption=None):
    p = fig_path(sub, name)
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{sub}/{name}.png` not found - run the pipeline first.")


def risk_band(pd_):
    if pd_ < 0.10:
        return "Low"
    if pd_ < 0.20:
        return "Moderate"
    if pd_ < 0.35:
        return "High"
    return "Very high"


def optimal_threshold(model_name, ratio, cals):
    """Threshold that minimises expected cost on the VALIDATION set
    (on calibrated probabilities)."""
    pv = load_table("val_predictions.parquet")
    p = pv[model_name].to_numpy()
    if model_name in cals:
        p = cals[model_name].predict(p)
    y = pv["y_true"].to_numpy()
    grid = np.arange(0.01, 0.99, 0.005)
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    cum_bad = np.cumsum(ys)                 # defaulters with PD <= cut (approved)
    cum_good = np.cumsum(1 - ys)
    idx = np.searchsorted(ps, grid, side="left")
    fn = np.where(idx > 0, cum_bad[np.maximum(idx - 1, 0)], 0)
    approved_good = np.where(idx > 0, cum_good[np.maximum(idx - 1, 0)], 0)
    fp = (1 - ys).sum() - approved_good
    cost = (ratio * fn + fp) / len(y)
    return float(grid[np.argmin(cost)])


def score(models, cals, X, name):
    """Calibrated probability of default."""
    p = models[name].predict_proba(X)
    if name in cals:
        p = cals[name].predict(p)
    return p


def plotly_layout(fig, height=420, **kw):
    layout = dict(height=height, margin=dict(l=10, r=10, t=40, b=10),
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                  legend=dict(font=dict(size=10)), hovermode="closest")
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# ==========================================================================
# Guard: pipeline must have been run
# ==========================================================================
models, cals = load_models()
if not models or not has("benchmark_results.csv"):
    st.error("No trained models found. Run `python run_all.py` from the project root first.")
    st.stop()
template, rate_by_sub, choices, qrange, spec = load_reference()
bench = load_table("benchmark_results.csv")
info = load_table("benchmark_info.json")
BEST = max(info, key=lambda m: info[m]["val_AUC"])

st.sidebar.title("📉 Loan Default Risk")
st.sidebar.caption("Lending Club 2007-2016 matured loans · 300k-borrower sample")
page = st.sidebar.radio("Navigate", [
    "Overview", "Score an applicant", "Batch scoring", "Model benchmark",
    "Lending policy simulator", "Analyses (A · C · F)", "Data & EDA"])
st.sidebar.divider()
st.sidebar.markdown("**Team**  \nKulshresth · Pranav Goyal · Aksha Alkesh Jain · "
                    "Abhinav Bhatia · Jadav Sai Rugvedh")


# ==========================================================================
# Overview
# ==========================================================================
if page == "Overview":
    st.title("Predicting loan defaults")
    st.write("We are the analytics team of a lending institution. Using **real Lending Club "
             "loan data**, we predict which applicants will default (charged-off) so that the "
             "lender can make better-informed approval decisions.")
    st.caption("**Scope:** the model scores loans Lending Club approved, after Lending Club had assigned a grade and interest rate. It has not been validated for rejected applicants or for decisions made before pricing.")
    summ = load_table("data_summary.json")
    b = bench.set_index("model")
    c = st.columns(5)
    c[0].metric("Raw loans", f"{summ['raw_rows']:,}")
    c[1].metric("Matured & resolved", f"{summ['matured_resolved_loans']:,}")
    c[2].metric("Modelling sample", f"{summ['sample_size']:,}")
    c[3].metric("Default rate", f"{summ['sample_default_rate']:.1%}")
    c[4].metric(f"Best test AUC ({BEST})", f"{b.loc[BEST, 'ROC_AUC']:.3f}")

    st.subheader("Pipeline")
    st.markdown("""
1. **Data**: 2.26M Lending Club loans (151 columns). We keep only loans that reached their
   scheduled maturity and have a final outcome: *Fully Paid* (0) vs *Charged Off / Default* (1).
2. **Leakage removal**: we drop 40 post-origination columns (payments, recoveries,
   settlement, hardship, refreshed FICO).
3. **Sampling**: a stratified 300k sample (issue-year × term × outcome), checked against the
   population with KS tests and a learning curve.
4. **Features**: 83 raw features (136 after one-hot encoding). Engineered features include
   affordability ratios, credit-history length and FICO midpoint. Missingness gets its own
   treatment: missing values become *ever happened* flags plus a cap, and a missing bureau
   block gets one flag.
5. **Models**: Logistic Regression, PCA + LR, Decision Tree, Random Forest, AdaBoost,
   Gradient Boosting, XGBoost, LightGBM and CatBoost. All use a 70/15/15 split, are tuned
   on validation and are reported on test.
6. **Analyses**: A - feature ablation, C - efficiency, F - explainability, plus
   cost-sensitive lending and calibration.
""")
    st.subheader("Leaderboard (test set)")
    cols = ["model", "family", "ROC_AUC", "PR_AUC", "KS", "Brier", "F1", "Recall", "Precision",
            "train_seconds"]
    st.dataframe(bench[cols].sort_values("ROC_AUC", ascending=False).style.format(
        {c: "{:.4f}" for c in cols[2:-1]} | {"train_seconds": "{:.1f}"})
        .background_gradient(subset=["ROC_AUC"], cmap="Blues"), hide_index=True, width="stretch")


# ==========================================================================
# Score an applicant
# ==========================================================================
elif page == "Score an applicant":
    st.title("Score a loan applicant")
    st.caption("Enter the application details. Bureau attributes not shown here are set to "
               "the typical (median) applicant from the training data. " + "**Scope:** the model scores loans Lending Club approved, after Lending Club had assigned a grade and interest rate. It has not been validated for rejected applicants or for decisions made before pricing.")

    sub_grades = [g + str(i) for g in "ABCDEFG" for i in range(1, 6)]
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Loan**")
            loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, 12000, step=500)
            term = st.radio("Term", [36, 60], horizontal=True, format_func=lambda t: f"{t} months")
            sg = st.selectbox("LC sub-grade", sub_grades, index=sub_grades.index("B4"))
            sgn = sub_grades.index(sg) + 1
            int_rate = st.number_input("Interest rate (%)", 5.0, 31.0,
                                       float(round(rate_by_sub.get(sgn, 12.0), 2)), step=0.25,
                                       help="Defaults to the median rate for the chosen sub-grade.")
            purpose = st.selectbox("Purpose", choices["purpose"],
                                   index=choices["purpose"].index("debt_consolidation"))
        with c2:
            st.markdown("**Borrower**")
            annual_inc = st.number_input("Annual income ($)", 5000, 1_000_000, 65000, step=1000)
            emp_len = st.select_slider("Employment length (years)",
                                       options=["n/a"] + [str(i) for i in range(0, 11)], value="5")
            emp_title = st.checkbox("Employment title provided", value=True)
            home = st.selectbox("Home ownership", choices["home_ownership"],
                                index=choices["home_ownership"].index("RENT")
                                if "RENT" in choices["home_ownership"] else 0)
            verif = st.selectbox("Income verification", choices["verification_status"])
            state = st.selectbox("State", choices["addr_state"],
                                 index=choices["addr_state"].index("CA")
                                 if "CA" in choices["addr_state"] else 0)
            dti = st.slider("Debt-to-income (%)", 0.0, 45.0, 17.0, 0.5)
        with c3:
            st.markdown("**Credit file**")
            fico = st.slider("FICO score", 660, 850, 700, 5)
            hist_years = st.slider("Credit history (years)", 1, 50, 15)
            revol_util = st.slider("Revolving utilisation (%)", 0.0, 120.0, 50.0, 1.0)
            revol_bal = st.number_input("Revolving balance ($)", 0, 500_000, 14000, step=500)
            inq = st.number_input("Credit inquiries, last 6 months", 0, 10, 0)
            delinq = st.number_input("Delinquencies, last 2 years", 0, 20, 0)
            cc = st.columns(2)
            open_acc = cc[0].number_input("Open accounts", 1, 60, 11)
            total_acc = cc[1].number_input("Total accounts", 2, 120, 24)
            pub_rec = cc[0].number_input("Public records", 0, 20, 0)
            mort_acc = cc[1].number_input("Mortgage accounts", 0, 20, 1)

    r = int_rate / 1200
    installment = loan_amnt * r / (1 - (1 + r) ** -term)

    row = dict(template)
    row.update({
        "loan_amnt": loan_amnt, "term_months": term, "int_rate": int_rate,
        "installment": installment, "grade_num": float("ABCDEFG".index(sg[0]) + 1),
        "sub_grade_num": float(sgn), "purpose": purpose, "annual_inc": annual_inc,
        "emp_length_num": np.nan if emp_len == "n/a" else float(emp_len),
        "emp_length_missing": int(emp_len == "n/a"), "emp_title_missing": int(not emp_title),
        "home_ownership": home, "verification_status": verif, "addr_state": state,
        "dti": dti, "fico_score": float(fico), "credit_history_months": hist_years * 12.0,
        "revol_util": revol_util, "revol_bal": revol_bal, "inq_last_6mths": inq,
        "delinq_2yrs": delinq, "open_acc": open_acc, "total_acc": total_acc,
        "pub_rec": pub_rec, "mort_acc": mort_acc,
        "loan_to_income": min(loan_amnt / annual_inc, 5),
        "installment_to_income": min(installment * 12 / annual_inc, 2),
        "revol_bal_to_income": min(revol_bal / annual_inc, 10),
    })
    if delinq > 0 and "ever_last_delinq" in row:
        row["ever_last_delinq"] = 1
        row["mths_since_last_delinq"] = 12.0
    elif "ever_last_delinq" in row:
        row["ever_last_delinq"] = 0
        row["mths_since_last_delinq"] = 240.0
    X = pd.DataFrame([row])[spec["numeric"] + spec["categorical"]]

    st.divider()
    left, right = st.columns([1, 1.25])
    with left:
        model_name = st.selectbox("Model", list(models), index=list(models).index(BEST),
                                  help="All probabilities are isotonic-calibrated on the "
                                       "validation set, so they are comparable across models.")
        ratio = st.slider("Cost of approving a defaulter vs declining a good borrower (FN : FP)",
                          1.0, 20.0, 4.0, 0.5)
        thr = optimal_threshold(model_name, ratio, cals)
        p = float(score(models, cals, X, model_name)[0])
        band = risk_band(p)
        icon, col = STATUS[band]
        decision = "DECLINE" if p >= thr else "APPROVE"
        m1, m2 = st.columns([1, 1.3])
        m1.metric("Probability of default", f"{p:.1%}")
        m2.markdown(
            f"<div style='padding-top:4px'><div style='font-size:0.85rem;color:{INK_2}'>Risk band"
            f"</div><div style='font-size:1.35rem;font-weight:600'>{icon} {band} risk</div>"
            f"<div style='font-size:0.85rem;color:{INK_2};margin-top:6px'>Decision</div>"
            f"<div style='font-size:1.35rem;font-weight:600'>"
            f"{'❌ Decline' if decision == 'DECLINE' else '✅ Approve'}</div></div>",
            unsafe_allow_html=True)
        st.caption(f"Cost-optimal cut-off at {ratio:g}:1 = **{thr:.1%}** PD (chosen on the "
                   f"validation set). Monthly instalment: **${installment:,.0f}**.")

        allp = {n: float(score(models, cals, X, n)[0]) for n in models}
        fig = go.Figure(go.Bar(
            x=[allp[n] * 100 for n in models], y=list(models), orientation="h",
            marker_color=[MODEL_COLORS[n] for n in models],
            text=[f"{allp[n]:.1%}" for n in models], textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.add_vline(x=thr * 100, line_dash="dot", line_color=MUTED,
                      annotation_text="cut-off", annotation_font_color=INK_2)
        plotly_layout(fig, 360, title="Predicted PD by model", xaxis_title="PD (%)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown(f"**Why?** Feature contributions for **{model_name}**")
        lm = models[model_name]
        contrib = None
        try:
            import shap
            Z = lm.transform(X)
            if model_name in ("Logistic Regression", "PCA + Logistic Regression"):
                ref = lm.transform(pd.DataFrame([template])[spec["numeric"] + spec["categorical"]])
                pipe = lm.estimator
                if model_name == "Logistic Regression":
                    zs = pipe.named_steps["scale"].transform(Z)[0]
                    zr = pipe.named_steps["scale"].transform(ref)[0]
                    vals = pipe.named_steps["clf"].coef_[0] * (zs - zr)
                else:
                    raise ValueError("no per-feature view for PCA components")
            elif model_name in ("AdaBoost", "Random Forest"):
                raise ValueError("TreeSHAP too slow / unsupported for this model in real time")
            else:
                sv = shap.TreeExplainer(lm.estimator)(Z).values
                vals = sv[0, :, 1] if sv.ndim == 3 else sv[0]
            s = pd.Series(vals, index=lm.feature_names)
            s.index = [onehot_source(f, spec["categorical"]) for f in s.index]
            contrib = s.groupby(level=0).sum()
        except Exception as e:  # noqa: BLE001
            st.info(f"Per-applicant explanation not available for {model_name} "
                    f"({e}). Choose XGBoost, LightGBM, CatBoost, Gradient Boosting, "
                    "Decision Tree or Logistic Regression.")
        if contrib is not None:
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:12])[::-1]
            fig = go.Figure(go.Bar(
                x=top.values, y=[nice(i) for i in top.index], orientation="h",
                marker_color=[SLOTS[1] if v > 0 else SLOTS[0] for v in top.values],
                hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
            plotly_layout(fig, 420, xaxis_title="Effect on log-odds of default "
                          "(orange = raises risk)")
            st.plotly_chart(fig, width="stretch")

        st.markdown("**What-if:** vary one input and keep the rest fixed")
        wf = st.selectbox("Feature", ["int_rate", "fico_score", "dti", "annual_inc", "loan_amnt",
                                      "revol_util", "credit_history_months", "inq_last_6mths"],
                          format_func=nice)
        lo, hi = qrange[wf]
        grid = np.linspace(lo, hi, 40)
        Xg = pd.concat([X] * len(grid), ignore_index=True)
        Xg[wf] = grid
        if wf in ("annual_inc", "loan_amnt"):
            Xg["loan_to_income"] = (Xg["loan_amnt"] / Xg["annual_inc"]).clip(upper=5)
            Xg["installment_to_income"] = (Xg["installment"] * 12 / Xg["annual_inc"]).clip(upper=2)
        pg = score(models, cals, Xg, model_name)
        fig = go.Figure(go.Scatter(x=grid, y=pg * 100, mode="lines",
                                   line=dict(color=MODEL_COLORS[model_name], width=2),
                                   hovertemplate=f"{nice(wf)}=%{{x:.1f}}<br>PD=%{{y:.1f}}%<extra></extra>"))
        fig.add_hline(y=thr * 100, line_dash="dot", line_color=MUTED,
                      annotation_text="cut-off", annotation_font_color=INK_2)
        fig.add_vline(x=float(X[wf].iloc[0]), line_color=SLOTS[1], line_width=1,
                      annotation_text="this applicant", annotation_font_color=INK_2)
        plotly_layout(fig, 320, xaxis_title=nice(wf), yaxis_title="PD (%)")
        st.plotly_chart(fig, width="stretch")


# ==========================================================================
# Batch scoring
# ==========================================================================
elif page == "Batch scoring":
    st.title("Batch scoring")
    st.write("Upload a CSV with applicants in the **original Lending Club column format**. "
             "**Required:** `loan_amnt`, `annual_inc`. **Recommended:** `term`, `int_rate`, "
             "`installment`, `grade`, `sub_grade`, `emp_length`, `home_ownership`, `purpose`, "
             "`dti`, `fico_range_low`, `fico_range_high`, `earliest_cr_line`, `issue_d`, "
             "`revol_bal`, `revol_util` and the other credit-bureau fields. If `installment` is "
             "missing it is computed from amount, rate and term. Any other missing input is set "
             "to the typical training value, and the app lists which ones. Post-loan columns, "
             "if present, are ignored.")

    sample = pd.read_parquet(SAMPLE_FILE)
    split = json.loads((ROOT / "data/processed/split_indices.json").read_text())
    demo = sample.loc[split["test"][:200]].drop(columns=["default"])
    demo["issue_d"] = demo["issue_d"].dt.strftime("%b-%Y")
    st.download_button("Download a 200-row example file (from the test set)",
                       demo.to_csv().encode(), "example_applicants.csv", "text/csv")

    up = st.file_uploader("CSV file", type=["csv"])
    model_sel = st.multiselect("Models", list(models), default=[BEST])
    ratio = st.slider("Cost ratio (FN : FP) for the decision", 1.0, 20.0, 4.0, 0.5)
    if up is not None and model_sel:
        raw = pd.read_csv(up, low_memory=False)
        X, filled, missing = prepare_applicants(raw, spec, template)
        if missing:
            st.error("The file is missing required column(s): "
                     + ", ".join(f"`{c}`" for c in missing)
                     + f". Required: {', '.join(REQUIRED_APPLICANT_COLS)}.")
            st.stop()
        if filled:
            n_all = len(spec["numeric"]) + len(spec["categorical"])
            st.warning(f"{len(filled)} of {n_all} model inputs were not in the file and were set "
                       "to the typical training value. Predictions are less specific to each "
                       "applicant when many inputs are filled.")
            with st.expander("Inputs filled with typical values"):
                st.write(", ".join(f"{nice(c)} (`{c}`)" for c in filled))
        out = raw.copy()
        for n in model_sel:
            p = score(models, cals, X, n)
            t = optimal_threshold(n, ratio, cals)
            out[f"PD_{slug(n)}"] = p.round(4)
            out[f"decision_{slug(n)}"] = np.where(p >= t, "DECLINE", "APPROVE")
        main_pd = out[f"PD_{slug(model_sel[0])}"]
        out["risk_band"] = main_pd.apply(risk_band)
        c = st.columns(3)
        c[0].metric("Applicants scored", f"{len(out):,}")
        c[1].metric("Mean PD", f"{main_pd.mean():.1%}")
        c[2].metric("Approval rate", f"{(out[f'decision_{slug(model_sel[0])}'] == 'APPROVE').mean():.0%}")
        fig = go.Figure(go.Histogram(x=main_pd * 100, nbinsx=40, marker_color=SLOTS[0]))
        plotly_layout(fig, 300, title=f"PD distribution ({model_sel[0]})", xaxis_title="PD (%)",
                      yaxis_title="Applicants")
        st.plotly_chart(fig, width="stretch")
        show = [c for c in ["loan_amnt", "term", "grade", "annual_inc", "risk_band"] if c in out]
        st.dataframe(out[show + [c for c in out.columns if c.startswith(("PD_", "decision_"))]],
                     width="stretch")
        st.download_button("Download scored file", out.to_csv(index=False).encode(),
                           "scored_applicants.csv", "text/csv")


# ==========================================================================
# Model benchmark
# ==========================================================================
elif page == "Model benchmark":
    st.title("Model benchmark")
    pt = load_table("test_predictions.parquet")
    y = pt["y_true"].to_numpy()
    tabs = st.tabs(["Leaderboard", "ROC & PR curves", "Threshold explorer", "Tuning log"])

    with tabs[0]:
        metric = st.selectbox("Metric", ["ROC_AUC", "PR_AUC", "KS", "Gini", "F1", "Recall",
                                         "Precision", "Balanced_Acc", "Brier", "LogLoss"])
        asc = metric in ("Brier", "LogLoss")
        b = bench.sort_values(metric, ascending=not asc)
        fig = go.Figure(go.Bar(x=b[metric], y=b["model"], orientation="h",
                               marker_color=[MODEL_COLORS[m] for m in b["model"]],
                               text=b[metric].round(4), textposition="outside"))
        plotly_layout(fig, 380, title=f"{metric} on the test set")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(bench.drop(columns=["TP", "FP", "TN", "FN"]).round(4), hide_index=True,
                     width="stretch")
        c = st.columns(2)
        with c[0]:
            show_fig("benchmark", "calibration_curves", "Reliability diagram")
        with c[1]:
            show_fig("benchmark", "confusion_matrices", "Confusion matrices")

    with tabs[1]:
        sel = st.multiselect("Models", MODEL_ORDER, default=MODEL_ORDER)
        c = st.columns(2)
        roc, prc = go.Figure(), go.Figure()
        for m in sel:
            fpr, tpr, _ = roc_curve(y, pt[m])
            k = max(1, len(fpr) // 600)
            roc.add_trace(go.Scatter(x=fpr[::k], y=tpr[::k], name=f"{m} ({roc_auc_score(y, pt[m]):.3f})",
                                     line=dict(color=MODEL_COLORS[m], width=2,
                                               dash="dash" if m == "Logistic Regression" else None)))
            pr, rc, _ = precision_recall_curve(y, pt[m])
            k = max(1, len(pr) // 600)
            prc.add_trace(go.Scatter(x=rc[::k], y=pr[::k], name=f"{m} ({average_precision_score(y, pt[m]):.3f})",
                                     line=dict(color=MODEL_COLORS[m], width=2,
                                               dash="dash" if m == "Logistic Regression" else None)))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash="dot", width=1),
                                 showlegend=False, hoverinfo="skip"))
        plotly_layout(roc, 480, title="ROC curve", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", hovermode="x unified")
        plotly_layout(prc, 480, title="Precision-recall curve", xaxis_title="Recall",
                      yaxis_title="Precision", hovermode="x unified")
        c[0].plotly_chart(roc, width="stretch")
        c[1].plotly_chart(prc, width="stretch")

    with tabs[2]:
        m = st.selectbox("Model", MODEL_ORDER, index=MODEL_ORDER.index(BEST))
        t = st.slider("Decision threshold (flag as default if PD >=)", 0.02, 0.80,
                      float(round(info[m]["threshold"], 2)), 0.01)
        yhat = (pt[m].to_numpy() >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
        c = st.columns(5)
        c[0].metric("Recall (defaulters caught)", f"{tp / (tp + fn):.1%}")
        c[1].metric("Precision", f"{tp / max(tp + fp, 1):.1%}")
        c[2].metric("Good borrowers declined", f"{fp / (fp + tn):.1%}")
        c[3].metric("F1", f"{2 * tp / max(2 * tp + fp + fn, 1):.3f}")
        c[4].metric("Approval rate", f"{(1 - yhat).mean():.1%}")
        cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual: paid", "Actual: default"],
                          columns=["Pred: paid", "Pred: default"])
        st.dataframe(cm.style.format("{:,}").background_gradient(cmap="Blues"), width="content")
        fig = go.Figure()
        for k, lab, col in ((0, "Fully paid", SLOTS[0]), (1, "Defaulted", SLOTS[1])):
            fig.add_trace(go.Histogram(x=pt.loc[y == k, m], name=lab, marker_color=col,
                                       opacity=0.6, nbinsx=60, histnorm="probability density"))
        fig.add_vline(x=t, line_dash="dot", line_color=INK_2)
        plotly_layout(fig, 340, barmode="overlay", title="Score distribution by actual outcome",
                      xaxis_title="Predicted PD")
        st.plotly_chart(fig, width="stretch")

    with tabs[3]:
        if has("tuning_results.csv"):
            st.dataframe(load_table("tuning_results.csv"), hide_index=True, width="stretch")
        st.json(info, expanded=False)


# ==========================================================================
# Lending policy simulator
# ==========================================================================
elif page == "Lending policy simulator":
    st.title("Cost-sensitive lending policy")
    cs = load_table("cost_summary.json") if has("cost_summary.json") else None
    pt = load_table("test_predictions.parquet")
    outc = pd.read_parquet(OUTCOME_FILE)
    profit = (outc["total_pymnt"] - outc["collection_recovery_fee"] - outc["loan_amnt"]).loc[pt.index].to_numpy()
    y = pt["y_true"].to_numpy()

    if cs:
        c = st.columns(4)
        c[0].metric("Avg loss per default", f"${cs['avg_loss_per_default']:,.0f}")
        c[1].metric("Avg profit per repaid loan", f"${cs['avg_profit_per_good']:,.0f}")
        c[2].metric("Empirical cost ratio", f"{cs['empirical_cost_ratio']:.1f} : 1")
        c[3].metric("Best model (validation AUC)", cs["best_model"])

    m = st.selectbox("Model", MODEL_ORDER, index=MODEL_ORDER.index(BEST))
    rule = st.radio("Decision rule", ["Single PD cut-off (cost ratio)",
                                      "Loan-level expected value (break-even PD by grade & term)"],
                    horizontal=True)
    p = pt[m].to_numpy()
    if m in cals:
        p = cals[m].predict(p)
    if rule.startswith("Single"):
        ratio = st.slider("Cost ratio FN : FP", 1.0, 30.0,
                          float(round(cs["empirical_cost_ratio"], 1)) if cs else 4.0, 0.5)
        t_opt = optimal_threshold(m, ratio, cals)
        manual = st.checkbox("Override the threshold manually")
        t = st.slider("PD cut-off", 0.01, 0.90, t_opt, 0.005) if manual else t_opt
        approve = p < t
    else:
        econ = load_table("loan_economics_by_grade_term.csv")
        econ = econ[econ["n_loans"] >= 200].set_index(["grade", "term_m"])["breakeven_pd"]
        o = outc.loc[pt.index]
        key = pd.MultiIndex.from_arrays([o["grade"].astype(str),
                                         o["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)])
        be = econ.reindex(key).fillna(0.31).to_numpy()
        st.caption("Approve when (1 - PD) x expected gain > PD x expected loss for the loan's "
                   "grade and term, i.e. PD < gain / (gain + loss). Economics come from training loans.")
        st.dataframe(load_table("loan_economics_by_grade_term.csv").round(3), hide_index=True,
                     width="stretch", height=220)
        ratio = cs["empirical_cost_ratio"] if cs else 2.2
        t = float(np.average(be))
        approve = p < be
    c = st.columns(5)
    c[0].metric("PD cut-off" if rule.startswith("Single") else "Average break-even PD", f"{t:.1%}")
    c[1].metric("Approval rate", f"{approve.mean():.1%}")
    c[2].metric("Default rate of approved", f"{y[approve].mean():.1%}",
                delta=f"{(y[approve].mean() - y.mean()) * 100:+.1f} pp vs approve-all",
                delta_color="inverse")
    c[3].metric("Net profit (test, $M)", f"{profit[approve].sum() / 1e6:.2f}",
                delta=f"{(profit[approve].sum() - profit.sum()) / 1e6:+.2f} vs approve-all")
    c[4].metric("Defaulters avoided", f"{((~approve) & (y == 1)).sum() / (y == 1).sum():.0%}")

    grid = np.arange(0.02, 0.8, 0.005)
    order = np.argsort(p)
    ps, prof_s, ys = p[order], profit[order], y[order]
    idx = np.searchsorted(ps, grid)
    cum_p = np.concatenate([[0], np.cumsum(prof_s)])[idx]
    cum_bad = np.concatenate([[0], np.cumsum(ys)])[idx]
    cum_good = idx - cum_bad
    cost = (ratio * cum_bad + ((1 - ys).sum() - cum_good)) / len(y)
    cc = st.columns(2)
    f1 = go.Figure(go.Scatter(x=grid, y=cost, line=dict(color=SLOTS[0], width=2),
                              hovertemplate="cut-off %{x:.3f}<br>cost %{y:.4f}<extra></extra>"))
    f1.add_vline(x=t, line_dash="dot", line_color=SLOTS[1])
    plotly_layout(f1, 340, title=f"Expected cost per applicant at {ratio:.1f}:1",
                  xaxis_title="PD cut-off", yaxis_title="Cost (FP = 1)", hovermode="x unified")
    cc[0].plotly_chart(f1, width="stretch")
    f2 = go.Figure(go.Scatter(x=grid, y=cum_p / 1e6, line=dict(color=SLOTS[2], width=2),
                              hovertemplate="cut-off %{x:.3f}<br>$%{y:.2f}M<extra></extra>"))
    f2.add_vline(x=t, line_dash="dot", line_color=SLOTS[1])
    f2.add_hline(y=profit.sum() / 1e6, line_dash="dash", line_color=MUTED,
                 annotation_text="approve everyone", annotation_font_color=INK_2)
    plotly_layout(f2, 340, title="Realised net profit on the test set", xaxis_title="PD cut-off",
                  yaxis_title="$ millions", hovermode="x unified")
    cc[1].plotly_chart(f2, width="stretch")

    if has("policy_comparison.csv"):
        st.subheader("Policies compared")
        pol = load_table("policy_comparison.csv")
        st.dataframe(pol.style.format({"approval_rate": "{:.1%}", "default_rate_approved": "{:.1%}",
                                       "defaulters_rejected_pct": "{:.1%}", "good_rejected_pct": "{:.1%}",
                                       "net_profit_usd": "${:,.0f}", "profit_per_applicant_usd": "${:,.0f}",
                                       "profit_vs_status_quo_usd": "${:+,.0f}"}),
                     hide_index=True, width="stretch")
    c = st.columns(2)
    with c[0]:
        show_fig("cost_sensitive", "threshold_vs_cost_ratio")
    with c[1]:
        show_fig("cost_sensitive", "profit_curve")
    memo = REPORTS_DIR / "memo_head_of_risk.md"
    if memo.exists():
        with st.expander("📄 Memo to the Head of Risk", expanded=False):
            st.markdown(memo.read_text())
            st.download_button("Download memo (.md)", memo.read_text(), "memo_head_of_risk.md")


# ==========================================================================
# Analyses
# ==========================================================================
elif page == "Analyses (A · C · F)":
    st.title("Additional analyses")
    ta, tc, tf = st.tabs(["A · Feature analysis & ablation", "C · Model efficiency",
                          "F · Interpretability & explainability"])
    with ta:
        if has("ablation_results.csv"):
            ab = load_table("ablation_results.csv")
            exp = st.selectbox("Experiment", ab["experiment"].unique().tolist())
            piv = ab[ab.experiment == exp].pivot(index="config", columns="model", values="ROC_AUC")
            st.dataframe(piv[[m for m in MODEL_ORDER if m in piv.columns]].style.format("{:.4f}")
                         .background_gradient(cmap="Blues", axis=None), width="stretch")
        sub = "analysis_A_ablation"
        c = st.columns(2)
        with c[0]:
            show_fig(sub, "leave_one_group_out")
            show_fig(sub, "feature_selection_topk")
        with c[1]:
            show_fig(sub, "single_group_only")
            show_fig(sub, "pca_components")
        show_fig(sub, "feature_engineering_value")
    with tc:
        if has("efficiency_summary.csv"):
            eff = load_table("efficiency_summary.csv")
            st.dataframe(eff[[c for c in ["model", "ROC_AUC", "train_seconds", "tuning_seconds",
                                          "inference_us_per_row", "single_prediction_ms",
                                          "model_size_mb", "n_trees", "n_nodes",
                                          "scaling_exponent"] if c in eff]].round(4),
                         hide_index=True, width="stretch")
        sub = "analysis_C_efficiency"
        show_fig(sub, "cost_bars")
        show_fig(sub, "tradeoff_pareto")
        show_fig(sub, "training_scaling")
    with tf:
        sub = "analysis_F_explainability"
        c = st.columns(2)
        with c[0]:
            show_fig(sub, "permutation_importance_heatmap")
            show_fig(sub, "importance_agreement")
            show_fig(sub, "shap_beeswarm_best")
        with c[1]:
            show_fig(sub, "group_importance")
            show_fig(sub, "shap_mean_abs_models")
            show_fig(sub, "logistic_coefficients")
        show_fig(sub, "partial_dependence_all_models")
        show_fig(sub, "ice_best_model")
        show_fig(sub, "shap_dependence_best")
        c = st.columns(3)
        for col, case in zip(c, ["high_risk_defaulter", "low_risk_repayer", "missed_default"]):
            with col:
                show_fig(sub, f"shap_waterfall_{case}")
        show_fig(sub, "decision_tree_top_levels")


# ==========================================================================
# Data & EDA
# ==========================================================================
elif page == "Data & EDA":
    st.title("Data, sampling & EDA")
    summ = load_table("data_summary.json")
    st.markdown(f"""
**Source:** Lending Club accepted loans 2007-2018Q4 (Kaggle `wordsforthewise/lending-club`):
{summ['raw_rows']:,} loans × {summ['raw_columns']} columns, one row per loan application.
Lending Club publishes no borrower ID (`member_id` is blank), so repeat borrowers are checked
separately (see *Repeat borrowers* below).

**Target:** `default = 1` if the loan was *Charged Off / Default*, `0` if *Fully Paid*.
Loans still *Current / Late / In grace period* have no final outcome and are excluded.

**Matured-vintage rule:** we keep only loans whose scheduled maturity date (issue date + term)
falls on or before the data snapshot (Mar-2019). This leaves {summ['matured_resolved_loans']:,}
loans with a default rate of {summ['population_default_rate']:.1%}.

**Sample:** a stratified random sample of {summ['sample_size']:,} loans (issue-year × term ×
outcome), split {summ['train']:,} / {summ['val']:,} / {summ['test']:,} (train / validation / test).
""")
    t = st.tabs(["Vintages & sampling", "Leakage", "Missing values", "Distributions",
                 "Correlations", "PCA"])
    with t[0]:
        show_fig("eda", "vintage_analysis")
        c = st.columns(2)
        with c[0]:
            show_fig("eda", "sampling_learning_curve")
        with c[1]:
            if has("sample_representativeness.csv"):
                st.markdown("**Sample vs population** (KS test for numeric, total-variation "
                            "distance for categorical)")
                st.dataframe(load_table("sample_representativeness.csv").round(4),
                             hide_index=True, width="stretch")
        if has("borrower_check.json"):
            bc = load_table("borrower_check.json")
            st.markdown("#### Repeat borrowers")
            st.markdown(
                f"`member_id` is filled in **{bc['member_id_non_missing']:,}** of "
                f"{bc['raw_loans']:,} loans, and the loan `id` is "
                f"{'unique' if bc['loan_id_unique'] else 'NOT unique'}. We matched loans on "
                "attributes that stay fixed for a person: first-credit month, 3-digit ZIP and "
                "job title, plus stated income for the strict key.")
            k = st.columns(3)
            k[0].metric("Repeat loans (strict)",
                        f"{bc['strict']['possible_repeat_pct']:.2%}")
            k[1].metric("Upper bound (loose)", f"{bc['loose']['possible_repeat_pct']:.2%}")
            k[2].metric("Max test-AUC change", f"{bc['max_abs_auc_change']:.4f}")
            st.caption(f"{bc['strict']['test_loans_sharing_key_with_train_val']:,} (strict) / "
                       f"{bc['loose']['test_loans_sharing_key_with_train_val']:,} (loose) of the "
                       "45,000 test loans share a key with a training or validation loan. "
                       "Removing them leaves every model's test AUC essentially unchanged.")
            if has("borrower_check_auc.csv"):
                with st.expander("Test AUC with and without possible repeat borrowers"):
                    st.dataframe(load_table("borrower_check_auc.csv").round(4),
                                 hide_index=True, width="stretch")
    with t[1]:
        st.markdown("Columns recorded **after** the loan is issued would leak the outcome. "
                    "A single such column predicts default almost perfectly, so all of them "
                    f"({len(summ['leakage_columns_removed'])}) are removed:")
        show_fig("eda", "leakage_demo")
        st.code(", ".join(summ["leakage_columns_removed"]))
    with t[2]:
        c = st.columns(2)
        with c[0]:
            show_fig("eda", "missing_values")
        with c[1]:
            show_fig("eda", "missingness_informative")
            st.markdown("""
* `mths_since_*` empty → the event **never happened**. We add an `ever_*` flag and fill with
  240 months.
* Bureau block empty (loans before ~2012) → one `bureau_data_missing` flag plus the median.
* Missing employment length / title → a flag (these borrowers default more often) plus
  the median.
* Fields that are > 50 % missing (added by Lending Club only in late 2015, or joint
  applications) are dropped.""")
    with t[3]:
        show_fig("eda", "class_balance")
        show_fig("eda", "default_rate_by_category")
        show_fig("eda", "numeric_distributions")
        show_fig("eda", "default_rate_by_state")
    with t[4]:
        c = st.columns(2)
        with c[0]:
            show_fig("eda", "correlation_with_target")
        with c[1]:
            show_fig("eda", "correlation_matrix")
    with t[5]:
        show_fig("benchmark", "pca_explained_variance")
        c = st.columns(2)
        with c[0]:
            show_fig("benchmark", "pca_scatter_pc1_pc2")
        with c[1]:
            show_fig("benchmark", "pca_loadings")
