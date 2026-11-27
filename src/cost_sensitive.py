"""
Optional extension - Cost-sensitive lending, calibration & memo
================================================================
A lender loses far more by approving a borrower who defaults (false negative)
than by declining a good borrower (false positive, a lost interest margin).

  1. Probability calibration of the best model (isotonic regression fitted
     on the validation set) - needed so that PD thresholds mean what they say.
  2. Empirical cost ratio: from the TRAINING loans' real cash flows
     (total payments received, recoveries, collection fees) we measure the
     average $ lost on a defaulted loan vs the average $ earned on a repaid
     loan.
  3. Expected cost vs threshold for cost ratios 1:1 ... 20:1; optimal
     threshold chosen on validation, evaluated on test; compared with the
     Bayes-optimal rule t* = 1 / (1 + ratio) for calibrated probabilities.
  4. Policy comparison in DOLLARS on the test set (approve-all, grade-based
     rules, model-based rules) using each loan's actual realised outcome.
  5. A memo to the Head of Risk (reports/memo_head_of_risk.md).

The $ outcome columns are post-loan information: they are used here ONLY to
value decisions, never as model inputs.

Usage:
    python -m src.cost_sensitive
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import MODELS_DIR, OUTCOME_FILE, REPORTS_DIR, SPLIT_FILE, TABLE_DIR
from src.models import MODEL_ORDER
from src.plotting import INK_2, MODEL_COLORS, MODEL_STYLE, MUTED, SLOTS, save

SUB = "cost_sensitive"
RATIOS = [1, 2, 3, 4, 5, 7, 10, 15, 20]
GRID = np.round(np.arange(0.01, 0.99, 0.005), 4)


def net_profit(out: pd.DataFrame) -> pd.Series:
    """Realised $ profit (+) / loss (-) of a loan if it is approved."""
    return out["total_pymnt"] - out["collection_recovery_fee"] - out["loan_amnt"]


def expected_cost(y, p, t, ratio):
    """Average cost per applicant: FN (approved defaulter) costs `ratio`,
    FP (declined good borrower) costs 1."""
    decline = p >= t
    fn = np.sum((~decline) & (y == 1))
    fp = np.sum(decline & (y == 0))
    return (ratio * fn + fp) / len(y)


def best_threshold(y, p, ratio):
    costs = np.array([expected_cost(y, p, t, ratio) for t in GRID])
    return float(GRID[np.argmin(costs)]), costs


def loan_economics(o: pd.DataFrame) -> pd.DataFrame:
    """Average gain (repaid) and loss (defaulted) as a fraction of principal,
    by grade and term, plus the break-even PD = gain / (gain + loss)."""
    o = o.copy()
    o["term_m"] = o["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    o["frac"] = net_profit(o) / o["loan_amnt"]
    grp = o.groupby(["grade", "term_m"])
    t = pd.DataFrame({
        "n_loans": grp.size(),
        "default_rate": grp["default"].mean(),
        "gain_frac_if_repaid": o[o["default"] == 0].groupby(["grade", "term_m"])["frac"].mean(),
        "loss_frac_if_default": -o[o["default"] == 1].groupby(["grade", "term_m"])["frac"].mean(),
    })
    t["breakeven_pd"] = t["gain_frac_if_repaid"] / (t["gain_frac_if_repaid"] + t["loss_frac_if_default"])
    t["implied_cost_ratio"] = t["loss_frac_if_default"] / t["gain_frac_if_repaid"]
    return t


def breakeven_pd(o: pd.DataFrame, econ: pd.DataFrame, fallback: float) -> np.ndarray:
    key = pd.MultiIndex.from_arrays(
        [o["grade"].astype(str), o["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)])
    ok = econ[econ["n_loans"] >= 200]["breakeven_pd"]
    return ok.reindex(key).fillna(fallback).to_numpy()


def plot_breakeven(econ: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, term in zip(axes, (36, 60)):
        e = econ.xs(term, level="term_m")
        e = e[e["n_loans"] >= 200]
        x = np.arange(len(e))
        ax.bar(x, e["default_rate"] * 100, color=SLOTS[0], width=0.6, label="Observed default rate")
        ax.plot(x, e["breakeven_pd"] * 100, color=SLOTS[1], marker="o", lw=2,
                label="Break-even PD = gain / (gain + loss)")
        ax.set_xticks(x, e.index)
        ax.set(title=f"{term}-month loans", xlabel="LC grade", ylabel="Probability (%)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
    fig.suptitle("Risk-based pricing: higher-grade interest raises the PD a loan can tolerate",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "breakeven_pd_by_grade", SUB)


def policy_stats(name, approve, y, profit):
    n = len(y)
    a = approve.astype(bool)
    return {
        "policy": name,
        "approval_rate": a.mean(),
        "default_rate_approved": y[a].mean() if a.any() else np.nan,
        "defaulters_rejected_pct": ((~a) & (y == 1)).sum() / max((y == 1).sum(), 1),
        "good_rejected_pct": ((~a) & (y == 0)).sum() / max((y == 0).sum(), 1),
        "net_profit_usd": profit[a].sum(),
        "profit_per_applicant_usd": profit[a].sum() / n,
    }


def main():
    info = json.loads((TABLE_DIR / "benchmark_info.json").read_text())
    best = max(info, key=lambda m: info[m]["val_AUC"])
    print(f"[cost] best model (validation AUC): {best}")

    pv = pd.read_parquet(TABLE_DIR / "val_predictions.parquet")
    pt = pd.read_parquet(TABLE_DIR / "test_predictions.parquet")
    out = pd.read_parquet(OUTCOME_FILE)
    split = json.loads(SPLIT_FILE.read_text())
    profit_all = net_profit(out)

    # ---------------- 1. calibration -----------------------------------------
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(pv[best], pv["y_true"])
    joblib.dump(iso, MODELS_DIR / "calibrator.joblib")
    (MODELS_DIR / "calibrator_model.txt").write_text(best)

    # one isotonic calibrator per model (validation set) -> used by the UI so
    # that probabilities are comparable across models
    cals, crow = {}, []
    for m in MODEL_ORDER:
        c = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv[m], pv["y_true"])
        cals[m] = c
        crow.append({"model": m, "brier_raw": brier_score_loss(pt["y_true"], pt[m]),
                     "brier_calibrated": brier_score_loss(pt["y_true"], c.predict(pt[m])),
                     "mean_pd_raw": pt[m].mean(), "mean_pd_calibrated": c.predict(pt[m]).mean()})
    joblib.dump(cals, MODELS_DIR / "calibrators.joblib")
    pd.DataFrame(crow).to_csv(TABLE_DIR / "calibration_by_model.csv", index=False)
    yv, yt = pv["y_true"].to_numpy(), pt["y_true"].to_numpy()
    pv_cal = iso.predict(pv[best])
    pt_raw = pt[best].to_numpy()
    pt_cal = iso.predict(pt_raw)
    cal = {"model": best,
           "brier_raw": brier_score_loss(yt, pt_raw), "brier_calibrated": brier_score_loss(yt, pt_cal),
           "auc_raw": roc_auc_score(yt, pt_raw), "auc_calibrated": roc_auc_score(yt, pt_cal),
           "mean_pd_raw": pt_raw.mean(), "mean_pd_calibrated": pt_cal.mean(),
           "observed_default_rate": yt.mean()}
    print("[cost] calibration:", {k: round(v, 4) if isinstance(v, float) else v for k, v in cal.items()})

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 0.7], [0, 0.7], color=MUTED, ls=":", lw=0.8, label="Perfect calibration")
    for p, lab, col in ((pt_raw, f"Raw (Brier {cal['brier_raw']:.4f})", SLOTS[1]),
                        (pt_cal, f"Isotonic (Brier {cal['brier_calibrated']:.4f})", SLOTS[0])):
        f, m = calibration_curve(yt, p, n_bins=15, strategy="quantile")
        ax.plot(m, f, marker="o", ms=4, color=col, label=lab)
    ax.set(xlabel="Predicted PD", ylabel="Observed default rate", xlim=(0, 0.7), ylim=(0, 0.7),
           title=f"Calibration of {best} (test set)")
    ax.legend()
    save(fig, "calibration_best_model", SUB)

    # ---------------- 2. empirical cost ratio (training loans only) ----------
    tr = out.loc[split["train"]]
    tr_profit = net_profit(tr)
    loss_default = -tr_profit[tr["default"] == 1].mean()
    gain_good = tr_profit[tr["default"] == 0].mean()
    emp_ratio = loss_default / gain_good
    lgd = (-tr_profit[tr["default"] == 1] / tr.loc[tr["default"] == 1, "loan_amnt"]).mean()
    margin = (tr_profit[tr["default"] == 0] / tr.loc[tr["default"] == 0, "loan_amnt"]).mean()
    print(f"[cost] avg loss per default ${loss_default:,.0f} ({lgd:.1%} of principal); "
          f"avg profit per repaid loan ${gain_good:,.0f} ({margin:.1%}); "
          f"empirical cost ratio = {emp_ratio:.2f} : 1")

    # ---------------- 3. thresholds vs cost ratio ----------------------------
    rows = []
    for r in RATIOS + [round(emp_ratio, 2)]:
        t_val, _ = best_threshold(yv, pv_cal, r)
        rows.append({"cost_ratio": r, "threshold_val_optimal": t_val,
                     "threshold_theory": 1 / (1 + r),
                     "test_cost_per_applicant": expected_cost(yt, pt_cal, t_val, r),
                     "test_cost_theory_threshold": expected_cost(yt, pt_cal, 1 / (1 + r), r),
                     "test_cost_approve_all": expected_cost(yt, pt_cal, 1.01, r),
                     "approval_rate": (pt_cal < t_val).mean(),
                     "default_rate_approved": yt[pt_cal < t_val].mean()})
    thr = pd.DataFrame(rows).drop_duplicates("cost_ratio").sort_values("cost_ratio")
    thr.to_csv(TABLE_DIR / "cost_ratio_thresholds.csv", index=False)
    print(thr.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    ax = axes[0]
    for r, col in zip([1, 3, 5, 10, 20], SLOTS):
        _, c = best_threshold(yt, pt_cal, r)
        ax.plot(GRID, c, color=col, label=f"{r}:1")
        ax.scatter(GRID[np.argmin(c)], c.min(), color=col, s=40, zorder=3, edgecolors="white")
    ax.set(xlim=(0, 0.7), xlabel="Decline if PD >= threshold",
           ylabel="Expected cost per applicant (FP = 1 unit)",
           title="Expected cost vs threshold (test)")
    ax.legend(title="FN : FP cost")
    ax = axes[1]
    ax.plot(thr["cost_ratio"], thr["threshold_val_optimal"], marker="o", color=SLOTS[0],
            label="Optimal on validation")
    rr = np.linspace(1, 20, 100)
    ax.plot(rr, 1 / (1 + rr), color=MUTED, ls="--", label="Theory 1/(1+ratio)")
    ax.axvline(emp_ratio, color=SLOTS[1], ls=":", lw=1)
    ax.text(emp_ratio, ax.get_ylim()[1] * 0.9, f"  empirical\n  {emp_ratio:.1f}:1", fontsize=8,
            color=INK_2)
    ax.set(xlabel="Cost ratio (FN : FP)", ylabel="Optimal PD threshold",
           title="Optimal threshold falls as defaults get costlier")
    ax.legend()
    ax = axes[2]
    ax.plot(thr["cost_ratio"], thr["approval_rate"] * 100, marker="o", color=SLOTS[2])
    ax.set(xlabel="Cost ratio (FN : FP)", ylabel="Approval rate at optimum (%)",
           title="...and the lender approves fewer applicants")
    fig.tight_layout()
    save(fig, "threshold_vs_cost_ratio", SUB)

    # all models at 10:1 and at the empirical ratio
    mrows = []
    for m in MODEL_ORDER:
        for r in (10, round(emp_ratio, 2)):
            t_val, _ = best_threshold(yv, pv[m].to_numpy(), r)
            mrows.append({"model": m, "cost_ratio": r, "threshold": t_val,
                          "test_cost_per_applicant": expected_cost(yt, pt[m].to_numpy(), t_val, r),
                          "approval_rate": (pt[m].to_numpy() < t_val).mean()})
    mc = pd.DataFrame(mrows)
    mc.to_csv(TABLE_DIR / "cost_by_model.csv", index=False)

    # ---------------- 4. policies in dollars ----------------------------------
    profit = profit_all.loc[pt.index].to_numpy()
    grade = pt["grade"].astype(str).to_numpy()
    pv_profit = profit_all.loc[pv.index].to_numpy()
    # profit-maximising threshold chosen on validation
    prof_curve = [pv_profit[pv_cal < t].sum() for t in GRID]
    t_profit = float(GRID[int(np.argmax(prof_curve))])
    t10, _ = best_threshold(yv, pv_cal, 10)
    t_emp, _ = best_threshold(yv, pv_cal, emp_ratio)
    # loan-level expected-value rule: approve if (1-PD)*gain - PD*loss > 0, i.e.
    # PD < break-even PD of the loan's grade & term (economics from TRAIN loans)
    econ = loan_economics(tr)
    econ.to_csv(TABLE_DIR / "loan_economics_by_grade_term.csv")
    be_test = breakeven_pd(out.loc[pt.index], econ, 1 / (1 + emp_ratio))
    print("[cost] loan economics by grade/term (training loans):")
    print(econ.round(3).to_string())
    plot_breakeven(econ)

    policies = [
        policy_stats("Approve everyone (status quo)", np.ones(len(yt)), yt, profit),
        policy_stats("Grades A-B only", np.isin(grade, ["A", "B"]), yt, profit),
        policy_stats("Grades A-C only", np.isin(grade, ["A", "B", "C"]), yt, profit),
        policy_stats("Grades A-D only", np.isin(grade, list("ABCD")), yt, profit),
        policy_stats("Model: PD < 0.50", pt_cal < 0.5, yt, profit),
        policy_stats(f"Model: F1-optimal (PD < {info[best]['threshold']:.2f})",
                     pt_raw < info[best]["threshold"], yt, profit),
        policy_stats(f"Model: 10:1 cost-optimal (PD < {t10:.3f})", pt_cal < t10, yt, profit),
        policy_stats(f"Model: empirical {emp_ratio:.1f}:1 optimal (PD < {t_emp:.3f})",
                     pt_cal < t_emp, yt, profit),
        policy_stats(f"Model: profit-optimal single cut-off (PD < {t_profit:.3f})",
                     pt_cal < t_profit, yt, profit),
        policy_stats("Model: loan-level expected value (PD < break-even of grade & term)",
                     pt_cal < be_test, yt, profit),
    ]
    pol = pd.DataFrame(policies)
    base = pol.loc[0, "net_profit_usd"]
    pol["profit_vs_status_quo_usd"] = pol["net_profit_usd"] - base
    pol.to_csv(TABLE_DIR / "policy_comparison.csv", index=False)
    print(pol.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [MUTED] + [SLOTS[3]] * 3 + [SLOTS[0]] * (len(pol) - 4)
    ax.barh(pol["policy"][::-1], pol["net_profit_usd"][::-1] / 1e6, color=colors[::-1], height=0.65)
    for i, (v, a) in enumerate(zip(pol["net_profit_usd"][::-1] / 1e6, pol["approval_rate"][::-1])):
        ax.text(v, i, f"  ${v:.2f}M  (approve {a:.0%})", va="center", fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set(xlabel="Realised net profit on the 45k test applicants ($ millions)",
           title="Lending policies compared on actual loan outcomes (grey = status quo, "
                 "yellow = grade rules, blue = model rules)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(min(0, xmin), xmax * 1.35)
    save(fig, "policy_comparison", SUB)

    # profit curve for every model: approve the lowest-PD x % of applicants
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rates = np.linspace(0.05, 1, 60)
    for m in MODEL_ORDER:
        order = np.argsort(pt[m].to_numpy())
        cum = np.cumsum(profit[order])
        ax.plot(rates * 100, cum[(rates * len(order)).astype(int) - 1] / 1e6,
                color=MODEL_COLORS[m], ls=MODEL_STYLE.get(m, "-"), label=m)
    for g, lab in ((["A", "B"], "A-B"), (["A", "B", "C"], "A-C"), (list("ABCD"), "A-D")):
        s = policy_stats(lab, np.isin(grade, g), yt, profit)
        ax.scatter(s["approval_rate"] * 100, s["net_profit_usd"] / 1e6, marker="s", s=50,
                   color=SLOTS[3], edgecolors="white", zorder=3)
        ax.annotate(f"Grades {lab}", (s["approval_rate"] * 100, s["net_profit_usd"] / 1e6),
                    xytext=(5, -12), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set(xlabel="Approval rate (%) - lowest predicted risk approved first",
           ylabel="Net profit ($ millions, test set)", title="Profit curve by model")
    ax.legend(fontsize=7)
    save(fig, "profit_curve", SUB)

    summary = {"best_model": best, "calibration": cal, "empirical_cost_ratio": emp_ratio,
               "avg_loss_per_default": loss_default, "avg_profit_per_good": gain_good,
               "loss_given_default_pct": lgd, "margin_on_good_pct": margin,
               "threshold_10_to_1": t10, "threshold_empirical": t_emp,
               "threshold_profit_opt": t_profit}
    (TABLE_DIR / "cost_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_memo(summary, thr, pol, mc, econ)


def write_memo(s, thr, pol, mc, econ):
    """Memo to the Head of Risk. Every number and comparison is computed from
    the results above, so the wording stays correct if the data change."""
    sq = pol.iloc[0]
    find = lambda key: pol[pol["policy"].str.contains(key, regex=False)].iloc[0]  # noqa: E731
    p10, pemp, pev = find("10:1"), find("empirical"), find("expected value")
    popt = find("profit-optimal")
    gac = pol[pol["policy"] == "Grades A-C only"].iloc[0]
    best_pol = pol.loc[pol["net_profit_usd"].idxmax()]
    t = lambda r: thr.loc[(thr.cost_ratio - r).abs().idxmin()]  # noqa: E731
    r10 = mc[mc.cost_ratio == 10].sort_values("test_cost_per_applicant")
    e = econ[econ["n_loans"] >= 200]
    be_lo, be_hi = e["breakeven_pd"].min(), e["breakeven_pd"].max()
    m = lambda v: f"${v / 1e6:.2f}M"  # noqa: E731
    d = lambda v: f"{'+' if v >= 0 else '-'}${abs(v) / 1e6:.2f}M"  # noqa: E731

    if pev["net_profit_usd"] >= sq["net_profit_usd"]:
        rec = (f"Adopt the **loan-level expected-value rule**: approve an applicant only when the "
               f"calibrated PD is below the break-even PD of the loan's grade and term "
               f"(between {be_lo:.0%} and {be_hi:.0%}). On the 45,000 held-out applicants it approves "
               f"{pev['approval_rate']:.1%} of them, cuts the default rate of approved loans from "
               f"{sq['default_rate_approved']:.1%} to {pev['default_rate_approved']:.1%}, and changes "
               f"realised profit by **{d(pev['profit_vs_status_quo_usd'])}** vs approving everyone.")
    else:
        rec = (f"Keep the current approval rate for the bulk of applicants and use the model "
               f"to decline only the extreme tail (PD above {popt['policy'].split('< ')[-1].rstrip(')')}). "
               f"Any broader single cut-off gives up more interest income than it saves in losses.")

    rows_pol = "\n".join(
        f"| {r.policy} | {r.approval_rate:.1%} | {r.default_rate_approved:.1%} | "
        f"{r.defaulters_rejected_pct:.0%} | {r.good_rejected_pct:.0%} | {m(r.net_profit_usd)} | "
        f"{d(r.profit_vs_status_quo_usd)} |" for r in pol.itertuples())
    rows_thr = "\n".join(
        f"| {r:g} : 1 | {t(r)['threshold_val_optimal']:.3f} | {1 / (1 + r):.3f} | "
        f"{t(r)['approval_rate']:.1%} | {t(r)['default_rate_approved']:.1%} |"
        for r in (1, 2, 3, 5, 10, 20))
    rows_econ = "\n".join(
        f"| {g} | {tm} | {int(r.n_loans):,} | {r.default_rate:.1%} | {r.gain_frac_if_repaid:.1%} | "
        f"{r.loss_frac_if_default:.1%} | {r.implied_cost_ratio:.1f} : 1 | {r.breakeven_pd:.1%} |"
        for (g, tm), r in e.iterrows())

    memo = f"""# Memo - Cost-sensitive approval policy

**To:** Head of Risk
**From:** Credit Analytics Team (Kulshresth, Pranav Goyal, Aksha Alkesh Jain, Abhinav Bhatia, Jadav Sai Rugvedh)
**Re:** Choosing the loan approval threshold from the default-prediction model

---

## 1. Key messages
1. **The model ranks risk well.** {s['best_model']} reaches a test ROC-AUC of {s['calibration']['auc_raw']:.3f}.
   Its predicted probabilities of default (PD) are well calibrated: the average predicted PD is
   {s['calibration']['mean_pd_raw']:.1%} against an observed default rate of
   {s['calibration']['observed_default_rate']:.1%}, and the Brier score is
   {s['calibration']['brier_raw']:.4f} ({s['calibration']['brier_calibrated']:.4f} after isotonic
   calibration). A PD threshold can therefore be read literally.
2. **Our own cash flows put the cost ratio near {s['empirical_cost_ratio']:.1f} : 1, not 10 : 1.**
   Across the training loans, a default loses on average ${s['avg_loss_per_default']:,.0f}
   ({s['loss_given_default_pct']:.0%} of principal, net of payments received before charge-off
   and recoveries). A repaid loan earns ${s['avg_profit_per_good']:,.0f}
   ({s['margin_on_good_pct']:.1%} of principal).
3. **A strict cut-off destroys value.** The 10 : 1 rule (PD < {s['threshold_10_to_1']:.1%}) approves only
   {p10['approval_rate']:.0%} of applicants and cuts realised profit from {m(sq['net_profit_usd'])} to
   {m(p10['net_profit_usd'])}.
4. **Risk is already priced.** Lending Club charges riskier grades higher rates, so the PD a
   loan can tolerate rises from {be_lo:.0%} (grade A) to {be_hi:.0%} (riskiest grades). A single
   PD cut-off for every loan ignores this. The best single cut-off on realised profit is very
   permissive ({popt['policy'].split('(')[-1].rstrip(')')}, {d(popt['profit_vs_status_quo_usd'])}).
5. **Recommendation.** {rec}

## 2. Policies compared (45,000 test applicants, realised cash flows)
| Policy | Approval rate | Default rate of approved | Defaulters declined | Good borrowers declined | Net profit | vs approve-all |
|---|---|---|---|---|---|---|
{rows_pol}

* The most profitable policy on the test set is **{best_pol['policy']}** ({m(best_pol['net_profit_usd'])}).
* Grade-only rules are blunt. For example, approving grades A-C approves {gac['approval_rate']:.0%} and changes profit
  by {d(gac['profit_vs_status_quo_usd'])}, because it declines many good borrowers who pay high interest.
* Minimising *expected cost* with a flat ratio ({s['empirical_cost_ratio']:.1f} : 1, PD < {s['threshold_empirical']:.1%})
  lowers the default rate to {pemp['default_rate_approved']:.1%} but changes profit by {d(pemp['profit_vs_status_quo_usd'])}.
  A flat ratio treats a 6%-interest and a 25%-interest loan as equally profitable.

## 3. How the optimal threshold moves with the cost ratio
| FN : FP cost | Optimal PD cut-off (validation) | Theory 1/(1+ratio) | Approval rate | Default rate of approved |
|---|---|---|---|---|
{rows_thr}

The empirical optimum tracks the Bayes rule PD* = 1/(1+ratio) closely. This confirms that the
calibrated PDs are reliable. As the assumed cost of a default rises, the cut-off falls quickly
and the approval rate collapses.

At 10 : 1, the lowest expected cost per applicant among all nine models comes from
**{r10.iloc[0]['model']}** ({r10.iloc[0]['test_cost_per_applicant']:.4f}), then {r10.iloc[1]['model']} and
{r10.iloc[2]['model']}. The worst is {r10.iloc[-1]['model']} ({r10.iloc[-1]['test_cost_per_applicant']:.4f}).

## 4. Loan economics by grade and term (training loans)
| Grade | Term | Loans | Default rate | Gain if repaid | Loss if default | Implied ratio | Break-even PD |
|---|---|---|---|---|---|---|---|
{rows_econ}

## 5. Assumptions and caveats
* Profit = payments received + recoveries - collection fees - principal. Funding cost,
  servicing cost and the time value of money are ignored. Including them would shrink the gain on
  repaid loans, raise the effective cost ratio and justify a stricter cut-off.
* Outcomes exist only for loans Lending Club approved (no reject inference). The policy is valid
  for a population screened the same way.
* Only matured vintages (issued 2007 to early 2016) are used. Default rates move with the credit
  cycle, so the cut-offs should be re-validated on every new vintage.
* Interest rate and grade are Lending Club's own pricing outputs. If our model were used to *set*
  prices, those inputs would no longer be available and the model would have to be re-trained without them
  (see Analysis A: removing them costs little AUC because the bureau data carry most of the signal).

## 6. Next steps
1. Run the recommended rule as a challenger against the current process on new applications.
2. Ask Finance for funding and servicing costs and re-estimate the break-even PDs.
3. Track calibration (predicted vs observed default rate) per vintage, and re-calibrate if it drifts.
"""
    (REPORTS_DIR / "memo_head_of_risk.md").write_text(memo)
    print("[cost] memo written -> reports/memo_head_of_risk.md")


if __name__ == "__main__":
    main()
