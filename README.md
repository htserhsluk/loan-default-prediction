# Predicting Loan Defaults - Lending Club

**Team:** Kulshresth · Pranav Goyal · Aksha Alkesh Jain · Abhinav Bhatia · Jadav Sai Rugvedh

We act as the analytics team of a lending institution. We use real Lending Club loan data
to predict which approved applicants will **default** (be charged off), so the lender can
make better-informed approval decisions.

---

## 1. Dataset

| | |
|---|---|
| Source | Lending Club accepted loans 2007-2018Q4, Kaggle [`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Institution | Lending Club, a US peer-to-peer lender registered with the SEC (a genuine lender, not a toy or benchmark dataset) |
| Size | **2,260,701 loans × 151 columns**, one row per borrower / loan |
| Target | `default = 1` for *Charged Off / Default*, `0` for *Fully Paid* |

The dataset meets every requirement in the brief: at least 100k borrowers, 50+ features,
one row per borrower, and a real lender.

### Data-handling decisions (all justified in the code and figures)
1. **Only loans with a final outcome.** *Current / Late / In Grace Period* loans are
   excluded because their outcome is unknown.
2. **Matured-vintage rule.** We keep only loans whose scheduled maturity
   (`issue_d + term`) is on or before the data snapshot (Mar-2019). Recent loans that are
   already "resolved" are a biased subset made up of early defaults and early pre-payments:
   their default rate swings from 23% (2016-17) to 16% (2018). See
   `reports/figures/eda/vintage_analysis.png`. This leaves **779,026 loans** with a 15.2%
   default rate.
3. **Leakage removal.** We remove 40 columns that are only known after the loan is issued:
   payments, recoveries, collection fees, refreshed FICO, hardship and settlement fields,
   and funded amounts. `leakage_demo.png` shows that a single such column reaches an
   AUC of up to 0.9996, while the best genuine application variable reaches only 0.67.
   We also drop identifiers and free text (`id`, `url`, `desc`, `title`, `emp_title`,
   `zip_code`).
4. **Sampling.** We draw a stratified random sample of **300,000 borrowers**, stratified on
   issue-year × term × outcome. All KS tests against the population give p > 0.5, and the
   total-variation distance for categorical variables is ≤ 0.002
   (`sample_representativeness.csv`). A learning curve on the full population
   (`sampling_learning_curve.png`) shows diminishing returns: training on all 679k loans
   instead of 210k gains only +0.005 AUC for 3× the compute. The sample keeps the slow
   prescribed models (sklearn Gradient Boosting, AdaBoost), SHAP and the ~200-model ablation
   grid feasible on a laptop.
5. **Missing values are informative and are not handled blindly.**
   * `mths_since_<event>`: missing means the event never happened. We add an `ever_<event>`
     flag and fill with a 240-month cap.
   * The credit-bureau block was not collected before ~2012. We add one
     `bureau_data_missing` flag and impute the median.
   * Missing employment length or title: we add a flag (these borrowers default at
     21% vs 15%) and impute the median.
   * Fields that are > 50% missing are dropped. These were added by Lending Club only
     in late 2015, or belong to joint applications.
6. **Split.** 70 / 15 / 15 stratified train / validation / test. The validation set is used
   for tuning, early stopping, threshold choice and calibration. The test set is used
   only for reporting.

Final feature set: **83 raw features** (76 numeric, 7 categorical), which become
**136 model inputs** after one-hot encoding. Engineered features include FICO midpoint,
credit-history length, loan / income, instalment / income, revolving balance / income,
numeric grade / sub-grade / employment length, and the missingness flags.

---

## 2. Models benchmarked
| Technique | Implementation |
|---|---|
| PCA | `StandardScaler → PCA (n components tuned: 80/90/95% variance) → Logistic Regression`, compared against plain Logistic Regression. Also a full PCA analysis (scree plot, class separation, loadings) and PCA with 5-40 components for every model in Analysis A |
| Decision Tree | `DecisionTreeClassifier`, depth and min-leaf tuned on validation |
| Random Forest | 300 trees, depth / leaf size tuned |
| Boosting | **AdaBoost**, **Gradient Boosting** (sklearn), **XGBoost**, **LightGBM**, **CatBoost**, with early stopping on validation |

Metrics: ROC-AUC, Gini, PR-AUC, KS, Brier, log-loss, and at the F1-optimal threshold
(chosen on validation): accuracy, balanced accuracy, precision, recall, specificity, F1.

### Benchmark results (45,000-borrower test set)
| Model | ROC-AUC | PR-AUC | KS | Brier | F1 | Recall | Precision | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| **CatBoost** | **0.7102** | 0.2994 | 0.307 | 0.1191 | 0.366 | 0.601 | 0.263 | 63.4 |
| LightGBM | 0.7091 | **0.3011** | 0.307 | 0.1191 | 0.366 | 0.536 | 0.278 | 20.4 |
| XGBoost | 0.7090 | 0.2999 | 0.306 | 0.1192 | 0.367 | 0.611 | 0.262 | 22.5 |
| Gradient Boosting | 0.7047 | 0.2928 | 0.302 | 0.1197 | 0.361 | 0.600 | 0.259 | 95.7 |
| Random Forest | 0.7009 | 0.2900 | 0.296 | 0.1202 | 0.358 | 0.629 | 0.250 | 71.7 |
| Logistic Regression | 0.7008 | 0.2898 | 0.293 | 0.1201 | 0.358 | 0.587 | 0.257 | 1.0 |
| PCA (86 PCs = 95% var.) + LR | 0.6982 | 0.2851 | 0.291 | 0.1205 | 0.359 | 0.588 | 0.259 | 0.4 |
| AdaBoost | 0.6944 | 0.2826 | 0.289 | 0.1515 | 0.352 | 0.625 | 0.245 | 146.0 |
| Decision Tree (depth 6) | 0.6820 | 0.2640 | 0.265 | 0.1218 | 0.343 | 0.574 | 0.245 | 5.0 |

*Threshold metrics use the F1-optimal threshold chosen on the validation set. The default rate is 15.2%, so a random model has PR-AUC 0.152.*

**Findings**
* The three modern gradient-boosting libraries (CatBoost, LightGBM, XGBoost) come out on top
  and are practically tied (ΔAUC ≤ 0.001). Classic sklearn Gradient Boosting is close behind.
* Logistic Regression is a strong baseline (0.701). The default signal in application data is
  largely monotonic, so a single deep tree (0.682) and AdaBoost (0.694) lag behind.
* **PCA** needs 86 of 136 components to keep 95% of the variance, because the features are only
  weakly correlated. Projecting onto them slightly *hurts* the linear model (0.698 vs 0.701)
  because PCA is unsupervised: it keeps directions of high variance, not of high predictive
  power. PC1/PC2 show heavily overlapping classes (`pca_scatter_pc1_pc2.png`).
* AdaBoost's raw scores are squashed towards 0.5 (Brier 0.152, mean score 0.33 vs a 15% default
  rate). Isotonic calibration on the validation set fixes this (Brier 0.121). The UI therefore
  shows calibrated probabilities for every model.
* An AUC of ~0.71 is typical for leakage-free Lending Club models. Lending Club had already
  screened and priced these borrowers, so the remaining risk is hard to separate. Models
  reporting 0.9+ on this dataset almost always use post-origination columns (see the leakage
  demo).

**Sampling check:** LightGBM trained on 210k rows reaches a hold-out AUC of 0.708. Trained on
all 679k available rows it reaches 0.713. Tripling the data (and training time) gains only
+0.005 AUC, so the 300k sample is a sensible trade-off.

---

## 3. Additional analyses (3 of 6 chosen: A, C, F)

### A. Feature analysis & ablation (`src/analysis_ablation.py`, `reports/figures/analysis_A_ablation/`)
We ran 25 feature configurations × 9 models. Each configuration was trained on the same
stratified 30k subsample and evaluated on the full 45k test set.

| Removed group | Mean ΔAUC over 9 models (×100) |
|---|---|
| LC risk grade (grade, sub-grade, interest rate) | **-1.09** |
| Credit score & history | -0.50 |
| Loan characteristics (amount, term, purpose, ratios) | -0.47 |
| Borrower profile (income, employment, home, state, DTI) | -0.21 |
| Balances & utilisation | -0.18 |
| Delinquency & public records | -0.04 |
| Account counts | -0.04 |

* **LC's own grade is the single most informative group.** On its own it reaches AUC ≈ 0.674,
  about 97% of the full model. Removing it costs the boosting models only 0.7-1.0 AUC points,
  so the credit-bureau data contain most of the same information. The single Decision Tree
  depends on it most (-2.7 points).
* **Delinquency / public-record features add almost nothing on top of the rest** (AUC 0.51-0.52
  alone). Most borrowers have clean records, and LC had already declined the worst cases.
* **Feature selection:** the top 40 of 83 raw features (LightGBM gain on train) come within
  0.1-0.4 AUC points of the full set, and even the top 5 reach 0.674-0.688. The Decision Tree *improves* with fewer features
  (0.676 with 5 vs 0.669 with all) because it overfits less.
* **Dimensionality reduction:** with 40 principal components, Logistic Regression is almost
  unchanged (-0.1 points). Random Forest and the gradient-boosting models lose 0.7-1.0 points,
  because PCA rotates away the monotone, axis-aligned splits that trees exploit. With only 5
  components, every model except the Decision Tree loses 1.2-2.7 points.
* **Engineering value:** the effects are small (≤ 0.25 points). The linear models gain most
  from the ratio features (+0.14) and the missingness flags (+0.23). XGBoost, LightGBM and
  CatBoost largely learn these patterns themselves (≤ 0.07). sklearn Gradient Boosting still
  benefits from the flags (+0.19).

### C. Model efficiency (`src/analysis_efficiency.py`, `reports/figures/analysis_C_efficiency/`)
| Model | Test AUC | Train (s) | Tuning total (s) | Batch inference (µs/row) | 1 applicant (ms) | Size (MB) | Trees / nodes | Scaling exponent |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.701 | 1.0 | 1.8 | 0.26 | 0.26 | 0.005 | 137 coef. | 1.01 |
| PCA + LR | 0.698 | 0.4 | 1.3 | 0.40 | 0.30 | 0.05 | 86 PCs | 0.95 |
| Decision Tree | 0.682 | 5.0 | 62.7 | **0.08** | **0.07** | 0.01 | 1 / 125 | 1.08 |
| Random Forest | 0.701 | 71.7 | 150.6 | 11.0 | 26.6 | **93.8** | 300 / 1.17M | **1.42** |
| AdaBoost | 0.694 | **146.0** | 146.0 | 19.6 | 20.7 | 0.22 | 250 / 1,750 | 1.09 |
| Gradient Boosting | 0.705 | 95.7 | 95.7 | 5.0 | 0.24 | 0.33 | 240 / 3,578 | 1.02 |
| XGBoost | 0.709 | 22.5 | 22.5 | 0.95 | 0.23 | 1.7 | 573 / 30.5k | 1.10 |
| LightGBM | 0.709 | 20.4 | 20.4 | 8.4 | 0.58 | 2.3 | 651 / 39.7k | 0.50 |
| **CatBoost** | **0.710** | 63.4 | 63.4 | 0.62 | 0.17 | 1.3 | 1,059 / 134k | 0.62 |

*Train time is the benchmark fit on 210k rows on an 8-core laptop (early stopping included). Tuning total is the sum of all grid fits. The scaling exponent is the slope of log(train time) vs log(rows) over 10k-210k with fixed hyper-parameters. These are single-run timings, so expect ±20-50% noise (e.g. XGBoost's 210k point).*

* **CatBoost is on all three Pareto frontiers** (AUC vs training time, inference time and
  size). It is the most accurate model, is tiny (1.3 MB), and scores one applicant in
  0.17 ms thanks to its symmetric (oblivious) trees.
* **Random Forest is the least efficient model.** It is 94 MB (fully grown trees, 1.17M
  nodes), takes 27 ms to score a single applicant (parallel-dispatch overhead), and its
  training time grows super-linearly (exponent 1.42) because the trees get deeper with
  more data. On AUC it only matches the 5 KB Logistic Regression.
* **AdaBoost and sklearn Gradient Boosting** train slowest, because they are single-threaded
  and boost sequentially with exact splits. LightGBM / XGBoost reach higher AUC 4-7× faster.
  LightGBM and CatBoost scale sub-linearly because histogram binning amortises over rows.
* **Logistic Regression** gives 95.5% of CatBoost's discriminative power (Gini 0.40 vs 0.42)
  for 1/60 of the training time. It is the right choice when a transparent, near-zero-cost scorecard is
  needed.

### F. Interpretability & explainability (`src/analysis_explainability.py`, `reports/figures/analysis_F_explainability/`)
Methods: model-agnostic **permutation importance** for all 9 models, at both raw-feature and
feature-group level (8k test rows, 3 repeats). **SHAP** (TreeExplainer) for CatBoost /
LightGBM / XGBoost / Decision Tree: global beeswarm, dependence plots and 3 local waterfalls.
**Partial dependence** for 8 key inputs across all models (on calibrated PDs), plus ICE curves.
Glass-box views: the decision-tree rules and the logistic-regression coefficients.

* **What drives risk.** Every model ranks LC's **sub-grade / grade** first. The next most
  important inputs are **accounts opened in the last 24 months** (recent credit-seeking),
  **DTI**, **loan / income**, **FICO**, **term** and **home ownership**. Shuffling the whole
  LC-grade group costs 4.9-14 AUC points. Shuffling the delinquency / public-record group
  costs ≈ 0, which matches Analysis A.
* **Direction of effects (SHAP + PDP; consistent across the boosted models):** a 60-month
  term adds about +3-5 PD points. PD rises steadily with DTI (≈ 14% → 18.5% over 0-35%) and
  with loan-to-income and revolving utilisation. It falls with FICO (≈ 16.5% at 660 →
  14% at 780) and with income up to ≈ $100k. Renting raises risk and having a mortgage lowers
  it.
* **Correlated features share credit.** grade, sub-grade and interest rate carry almost the
  same information. Individually they look modest for Random Forest and CatBoost, but jointly
  they dominate. Holding sub-grade fixed, extra interest rate adds little for the boosters
  (flat PDP).
* **Model agreement.** The boosted models agree strongly on feature rankings (Spearman
  0.65-0.78). Logistic Regression, PCA + LR and the Decision Tree agree much less (0.17-0.48).
  The linear model leans on the engineered *loan / income* ratio, and the single tree on a
  handful of grade splits.
* **Local explanations (CatBoost waterfalls).** Three applicants are shown:
  * A *high-risk defaulter* with PD 62%: grade G3, 60-month term, 8 accounts opened in
    24 months, no bankcard limit, FICO 667.
  * A *low-risk repayer* with PD 0.8%: grade A1, FICO 837, DTI 6%, 2% revolving
    utilisation.
  * A *missed default* with PD 2.4%: grade A1, 5.3% rate, DTI 9%. On paper this applicant
    looks safe. Such defaults are driven by events after origination (job loss, illness)
    that no application-time model can see.
* **Caveat:** PDP / ICE change one input while holding correlated inputs fixed (e.g. interest
  rate without sub-grade), so the extremes of some curves (very high incomes) are
  extrapolations into sparse regions.

### Optional extension: cost-sensitive lending (memo: `reports/memo_head_of_risk.md`)
* **Calibration:** CatBoost probabilities are already well calibrated (mean PD 15.2% vs 15.2%
  observed). An isotonic calibrator is still fitted for every model.
* **Empirical cost ratio from real cash flows (training loans):** a default loses $4,996
  on average (38% of principal, net of payments before charge-off and recoveries). A repaid
  loan earns $2,283 (16.7%). That puts the cost ratio at **≈ 2.2 : 1**, far below the 10 : 1
  used as an example in the brief.
* The optimal threshold on validation tracks the Bayes rule PD* = 1/(1+ratio). It falls from
  0.43 at 1:1 to 0.09 at 10:1 and 0.045 at 20:1, and the approval rate falls from 99.6% to 34%
  to 11%.
* **Policies valued with each test loan's actual outcome:**

| Policy | Approval | Default rate of approved | Net profit (45k applicants) |
|---|---|---|---|
| Approve everyone (status quo) | 100% | 15.2% | $52.58M |
| Grades A-C only | 80% | 12.2% | $39.80M |
| Model, 10:1 cost-optimal (PD < 0.09) | 34% | 5.7% | $17.86M |
| Model, empirical 2.2:1 (PD < 0.31) | 91% | 13.1% | $49.72M |
| **Model, loan-level expected value** | **97%** | **14.4%** | **$52.99M (+$0.41M)** |

  Lending Club's interest rates already price risk: the break-even PD rises from about 20%
  for grade A to about 60% for grade G (60-month loans). So a single PD cut-off, or a
  grade-only rule, throws away profitable high-yield loans. The recommended rule approves a
  loan only when PD < gain / (gain + loss) for its grade and term. It is the only policy that
  beats approving everyone on realised profit.

---

## 4. User interface (`app/streamlit_app.py`)
`streamlit run app/streamlit_app.py`, integrated with the trained pipeline:
* **Score an applicant**: enter the loan, borrower and credit-file details. You get a live
  calibrated PD from all 9 models, a risk band, and an approve / decline decision at a chosen
  cost ratio. The page also shows a per-applicant explanation (SHAP or coefficient
  contributions) and a what-if curve for any input.
* **Batch scoring**: upload a CSV in the raw Lending Club format and download it with PDs and
  decisions added. An example file can be downloaded from the app.
* **Model benchmark**: leaderboard by any metric, interactive ROC / PR curves, a threshold
  explorer with a live confusion matrix and score distributions, and the tuning log.
* **Lending policy simulator**: choose a cost-ratio cut-off or the expected-value rule and see
  approval rate, default rate and realised $ profit. The Head-of-Risk memo is included.
* **Analyses** and **Data & EDA**: every figure and table from the pipeline.

---

## 5. Project structure
```
├── run_all.py                    # runs the whole pipeline end-to-end
├── requirements.txt
├── src/
│   ├── config.py                 # paths, seed, leakage column list, settings
│   ├── download_data.py          # Kaggle download
│   ├── data_prep.py              # target, matured vintages, leakage removal, sampling, split
│   ├── features.py               # feature engineering, missing-value strategy, preprocessor
│   ├── eda.py                    # exploratory data analysis
│   ├── models.py                 # model factory, metrics, LoanModel wrapper
│   ├── train_benchmark.py        # training, tuning, benchmarking, PCA analysis
│   ├── sampling_check.py         # learning curve on the full population
│   ├── analysis_ablation.py      # Analysis A
│   ├── analysis_efficiency.py    # Analysis C
│   ├── analysis_explainability.py# Analysis F
│   ├── cost_sensitive.py         # cost-sensitive extension + memo
│   └── plotting.py               # shared chart style / colours
├── app/streamlit_app.py          # user interface
├── reports/
│   ├── figures/                  # all figures (eda, benchmark, analysis_*, cost_sensitive)
│   ├── tables/                   # all result tables (csv / json)
│   ├── memo_head_of_risk.md
│   └── decision_tree_rules.txt
└── docs/                         # project brief and marking scheme
```

## 6. How to run
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Kaggle API token in ~/.kaggle/kaggle.json (Kaggle → Settings → Create New Token)
python run_all.py                  # ~45-60 min on an 8-core laptop
streamlit run app/streamlit_app.py # opens the UI at http://localhost:8501
```
Single steps can be re-run, e.g. `python run_all.py --only ablation` or
`python run_all.py --from explain`.

On macOS, XGBoost / LightGBM need OpenMP: `brew install libomp`.
