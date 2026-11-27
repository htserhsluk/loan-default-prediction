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
| Size | **2,260,668 loans × 151 columns**, one row per loan, reduced to one loan per borrower (item 3) |
| Target | `default = 1` for *Charged Off / Default*, `0` for *Fully Paid* |

The dataset meets the brief's requirements: a real lender, far more than 100k borrowers,
50+ features, and one row per borrower (not monthly payment records). Lending Club does not
publish a borrower ID (`member_id` is blank in all 2,260,668 rows), so we identify a borrower
by attributes that stay fixed between applications and keep only their first loan (item 3
below). Item 8 measures the repeat borrowers this cannot catch.

**Scope of the model.** It scores loans that Lending Club approved, at the point where Lending
Club had already assigned a grade and interest rate. The results do not cover rejected
applicants or decisions made before pricing. Without Lending Club's grade, sub-grade and rate,
the boosted models lose only 0.5-0.9 AUC points (Analysis A), so most of the signal comes from
the applicant's own credit file.

### Data-handling decisions (all justified in the code and figures)
1. **Only loans with a final outcome.** *Current / Late / In Grace Period* loans are
   excluded because their outcome is unknown.
2. **Matured-vintage rule.** We keep only loans whose scheduled maturity
   (`issue_d + term`) is on or before the data snapshot (Mar-2019). Recent loans that are
   already "resolved" are a biased subset made up of early defaults and early pre-payments:
   their default rate swings from 23% (2016-17) to 16% (2018). See
   `reports/figures/eda/vintage_analysis.png`. This leaves **779,026 loans**.
3. **One loan per borrower** (`one_loan_per_borrower` in `src/data_prep.py`). Loan `id` is
   unique, but there is no borrower ID. Loans that share first-credit month, 3-digit ZIP and
   job title (lower-cased) are treated as one borrower, and only that borrower's first loan
   (earliest issue date) is kept.
   * This removes **27,018** later loans (3.5%; their default rate is 14.1%) and leaves
     **752,008 loans**, one per identified borrower, with a 15.2% default rate.
   * The 51,449 loans with no job title cannot be matched, so they are kept.
   * The key also merges some different people who share a common job title, ZIP and
     first-credit month. Dropping their later loans costs data but does not create
     duplicates.
4. **Leakage removal.** We remove 40 columns that are only known after the loan is issued:
   payments, recoveries, collection fees, refreshed FICO, hardship and settlement fields,
   and funded amounts. `leakage_demo.png` shows that a single such column reaches an
   AUC of up to 0.9995, while the best genuine application variable reaches only 0.67.
   We also drop identifiers and free text (`id`, `url`, `desc`, `title`, `emp_title`,
   `zip_code`), after using the last two for item 3.
5. **Sampling.** We draw a stratified random sample of **300,000 loans**, stratified on
   issue-year × term × outcome. All KS tests against the population give p > 0.2, and the
   total-variation distance for categorical variables is ≤ 0.002
   (`sample_representativeness.csv`). A learning curve on the full population
   (`sampling_learning_curve.png`) shows diminishing returns: training on all 652k loans
   instead of 210k gains only +0.005 AUC for 3× the compute. The sample keeps the slow
   prescribed models (sklearn Gradient Boosting, AdaBoost), SHAP and the ~200-model ablation
   grid feasible on a laptop.
6. **Missing values are informative and are not handled blindly.**
   * `mths_since_<event>`: missing means the event never happened. We add an `ever_<event>`
     flag and fill with a 240-month cap.
   * The credit-bureau block was not collected before ~2012. We add one
     `bureau_data_missing` flag and impute the median.
   * Missing employment length or title: we add a flag (these borrowers default at
     21% vs 15%) and impute the median.
   * Fields that are > 50% missing are dropped. These were added by Lending Club only
     in late 2015, or belong to joint applications.
7. **Split.** 70 / 15 / 15 stratified train / validation / test. The validation set is used
   for tuning, early stopping, threshold choice and calibration. The test set is used
   only for reporting.
8. **Repeat borrowers left** (`src/borrower_check.py`). After item 3, no two loans share
   the borrower key. That key misses a repeat borrower who moved or changed jobs, so we
   searched again with two keys that each swap one part for the stated income:
   * *moved* (first-credit month + job title + income): **2.6%** of the 752k loans are
     possible repeats.
   * *new job* (first-credit month + 3-digit ZIP + income; also covers loans with no job
     title): **5.7%**.
   * Both keys also match different people who share these values, so these are
     estimates, not exact counts. No borrower ID exists to verify them.
   * 867 (moved) and 1,895 (new job) of the 45,000 test loans share such a key with a
     training or validation loan. After excluding them, every model's test ROC-AUC changed
     by **less than 0.001**, and every change was a small *increase*. Any repeat borrowers
     left therefore do not inflate the reported results (`borrower_check.json`,
     `borrower_check_auc.csv`).

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
| **CatBoost** | **0.7085** | **0.3018** | 0.304 | 0.1193 | 0.366 | 0.600 | 0.263 | 38.8 |
| LightGBM | 0.7077 | 0.2988 | 0.303 | 0.1195 | 0.369 | 0.578 | 0.271 | 19.2 |
| XGBoost | 0.7069 | 0.2999 | 0.303 | 0.1195 | 0.363 | 0.588 | 0.262 | 18.2 |
| Gradient Boosting | 0.7038 | 0.2951 | 0.297 | 0.1199 | 0.362 | 0.623 | 0.255 | 71.0 |
| Random Forest | 0.7011 | 0.2909 | 0.294 | 0.1204 | 0.360 | 0.597 | 0.258 | 47.6 |
| Logistic Regression | 0.7007 | 0.2901 | 0.294 | 0.1203 | 0.359 | 0.612 | 0.254 | 1.1 |
| PCA (87 PCs = 95% var.) + LR | 0.6969 | 0.2857 | 0.290 | 0.1207 | 0.359 | 0.593 | 0.258 | 0.6 |
| AdaBoost | 0.6944 | 0.2828 | 0.285 | 0.1519 | 0.353 | 0.614 | 0.248 | 142.7 |
| Decision Tree (depth 6) | 0.6818 | 0.2633 | 0.269 | 0.1220 | 0.344 | 0.666 | 0.232 | 4.7 |

*Threshold metrics use the F1-optimal threshold chosen on the validation set. The default rate is 15.2%, so a random model has PR-AUC 0.152.*

**Findings**
* The three modern gradient-boosting libraries (CatBoost, LightGBM, XGBoost) come out on top
  and are practically tied (ΔAUC ≤ 0.002). Classic sklearn Gradient Boosting is close behind.
* Logistic Regression is a strong baseline (0.701). The default signal in application data is
  largely monotonic, so a single deep tree (0.682) and AdaBoost (0.694) lag behind.
* **PCA** needs 87 of 136 components to keep 95% of the variance, because the features are only
  weakly correlated. Projecting onto them slightly *hurts* the linear model (0.697 vs 0.701)
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
all 652k available rows it reaches 0.713. Tripling the data (and training time) gains only
+0.005 AUC, so the 300k sample is a sensible trade-off.

---

## 3. Additional analyses (3 of 6 chosen: A, C, F)

### A. Feature analysis & ablation (`src/analysis_ablation.py`, `reports/figures/analysis_A_ablation/`)
We ran 25 feature configurations for all 9 models: 221 fits in total. In the 4
PCA-component configurations, PCA + LR is skipped because it would duplicate Logistic
Regression trained on the same components. Each fit used the same stratified 30k training
subsample and was evaluated on the full 45k test set.

| Removed group | Mean ΔAUC over 9 models (×100) |
|---|---|
| LC risk grade (grade, sub-grade, interest rate) | **-1.08** |
| Credit score & history | -0.79 |
| Loan characteristics (amount, term, purpose, ratios) | -0.62 |
| Borrower profile (income, employment, home, state, DTI) | -0.33 |
| Balances & utilisation | -0.14 |
| Delinquency & public records | -0.12 |
| Account counts | -0.07 |

* **LC's own grade is the single most informative group.** On its own it reaches AUC ≈ 0.67,
  about 97% of the full model. Removing it costs the boosting models only 0.5-0.9 AUC points,
  so the credit-bureau data contain most of the same information. The single Decision Tree
  depends on it most (-2.6 points).
* **Delinquency / public-record features add little on top of the rest** (AUC 0.52 alone).
  Most borrowers have clean records, and LC had already declined the worst cases.
* **Feature selection:** the top 40 of 83 raw features (LightGBM gain on train) come within
  0.3-0.8 AUC points of the full set, and even the top 5 reach 0.672-0.685. The Decision Tree
  does *better* with fewer features (0.674 with 5 and 0.677 with 10, vs 0.668 with all) because
  it overfits less.
* **Dimensionality reduction:** with 40 principal components, Logistic Regression loses only
  0.3 points. Random Forest and the gradient-boosting models lose 0.7-1.2 points, because PCA
  rotates away the monotone, axis-aligned splits that trees exploit. With only 5 components,
  every model except the Decision Tree loses 1.4-2.5 points.
* **Engineering value:** the effects are small (≤ 0.45 points) and partly noise at a 30k-row
  training subsample. The missingness flags help every model except the Decision Tree
  (+0.06 to +0.44, most for XGBoost and sklearn Gradient Boosting). The ratio features help
  most models a little (up to +0.23), while LightGBM and CatBoost do slightly better without
  them (by 0.1-0.15), because they build these interactions from the raw amounts themselves.

### C. Model efficiency (`src/analysis_efficiency.py`, `reports/figures/analysis_C_efficiency/`)
| Model | Test AUC | Train (s) | Tuning total (s) | Batch inference (µs/row) | 1 applicant, model only (ms) | Size (MB) | Trees / nodes | Scaling exponent |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.701 | 1.1 | 1.9 | 0.22 | 0.21 | 0.005 | 137 coef. | 1.11 |
| PCA + LR | 0.697 | 0.6 | 1.4 | 0.36 | 0.30 | 0.05 | 87 PCs | 1.05 |
| Decision Tree | 0.682 | 4.7 | 59.3 | **0.08** | **0.07** | 0.01 | 1 / 123 | 1.10 |
| Random Forest | 0.701 | 47.6 | 102.0 | 8.5 | 30.2 | **132.1** | 300 / 1.65M | **1.31** |
| AdaBoost | 0.694 | **142.7** | 142.7 | 17.6 | 19.6 | 0.22 | 250 / 1,750 | 1.09 |
| Gradient Boosting | 0.704 | 71.0 | 71.0 | 3.4 | 0.22 | 0.28 | 197 / 2,941 | 1.10 |
| XGBoost | 0.707 | 18.2 | 18.2 | 1.23 | 0.38 | 1.9 | 672 / 35.8k | 0.70 |
| LightGBM | 0.708 | 19.2 | 19.2 | 10.8 | 0.42 | 3.3 | 935 / 57.0k | 0.36 |
| **CatBoost** | **0.709** | 38.8 | 38.8 | 0.59 | 0.17 | 1.4 | 1,185 / 150k | 0.57 |

*Train time is the benchmark fit on 210k rows on an 8-core laptop (early stopping included). Tuning total is the sum of all grid fits. The scaling exponent is the slope of log(train time) vs log(rows) over 10k-210k with fixed hyper-parameters. Batch and single-applicant times cover the model step only, on already-preprocessed input. Preprocessing one applicant adds about 5-10 ms for every model, so end-to-end scoring of one applicant takes about 6 ms with CatBoost. These are single-run timings, so expect ±20-50% noise.*

* **CatBoost is on all three Pareto frontiers** (AUC vs training time, inference time and
  size). It is the most accurate model, is small (1.4 MB), and its model step scores one
  applicant in 0.17 ms thanks to its symmetric (oblivious) trees.
* **Random Forest is the least efficient model.** It is 132 MB (300 trees of depth 16,
  1.65M nodes), its model step takes 30 ms for a single applicant (parallel-dispatch
  overhead), and its training time grows super-linearly (exponent 1.31) because each tree
  grows more nodes as the data grow. On AUC it only matches the 5 KB Logistic Regression.
* **AdaBoost and sklearn Gradient Boosting** train slowest, because they are single-threaded
  and boost sequentially with exact splits. LightGBM / XGBoost reach higher AUC 4-8× faster.
  The histogram-based libraries (LightGBM, CatBoost, XGBoost) scale sub-linearly here
  (exponents 0.4-0.7) because binning amortises over rows.
* **Logistic Regression** gives 96% of CatBoost's discriminative power (Gini 0.40 vs 0.42)
  for 1/35 of the training time. It is the right choice when a transparent, near-zero-cost
  scorecard is needed.

### F. Interpretability & explainability (`src/analysis_explainability.py`, `reports/figures/analysis_F_explainability/`)
Methods: model-agnostic **permutation importance** for all 9 models, at both raw-feature and
feature-group level (8k test rows, 3 repeats). **SHAP** (TreeExplainer) for CatBoost /
LightGBM / XGBoost / Decision Tree: global beeswarm, dependence plots and 3 local waterfalls.
**Partial dependence** for 8 key inputs across all models (on calibrated PDs), plus ICE curves.
Glass-box views: the decision-tree rules and the logistic-regression coefficients.

* **What drives risk.** At group level, every model ranks LC's **risk grade** first. Among
  single inputs, **sub-grade / grade** lead for the boosted models and the Decision Tree, and
  **accounts opened in the last 24 months** (recent credit-seeking) is first or second for
  every tree ensemble. Logistic Regression leans most on **loan / income**. Other important
  inputs are **term**, **DTI**, **months since the most recent bankcard** and **total
  bankcard limit**. Shuffling the whole LC-grade group costs 4.4-14.5 AUC points. Shuffling
  the delinquency / public-record group costs under 0.5 points, which matches Analysis A.
* **Direction of effects (SHAP + PDP; consistent across the boosted models):** a 60-month
  term adds about +4-5 PD points. PD rises steadily with DTI (≈ 14% → 17.5% over 0-35%) and
  with loan-to-income and revolving utilisation. It falls with FICO (≈ 16% at 660 →
  13% at 780) and with income up to ≈ $100k. Renting raises risk and having a mortgage lowers
  it.
* **Correlated features share credit.** grade, sub-grade and interest rate carry almost the
  same information. Individually they look modest for Random Forest and CatBoost, but jointly
  they dominate. Holding sub-grade fixed, extra interest rate adds little for the boosters:
  their PDP is flat or slightly falling above a rate of about 10%.
* **Model agreement.** XGBoost, LightGBM and CatBoost agree strongly on feature rankings
  (Spearman 0.70-0.85), and moderately with Random Forest, AdaBoost and sklearn Gradient
  Boosting (0.55-0.64). Logistic Regression, PCA + LR and the Decision Tree agree much less
  (0.18-0.53).
  The linear model leans on the engineered *loan / income* ratio, and the single tree on a
  handful of grade splits.
* **Local explanations (CatBoost waterfalls).** Three applicants are shown:
  * A *high-risk defaulter* with PD 59%: grade E1, 60-month term, a bankcard opened a month
    earlier, no bankcard credit left, a loan worth 44% of income, FICO 707.
  * A *low-risk repayer* with PD 1.1%: grade A1, FICO 797, DTI 3.5%, 10% revolving
    utilisation.
  * A *missed default* with PD 2.4%: grade A1, 6.0% rate, FICO 782, DTI 17%. On paper this applicant
    looks safe. The data do not record why the loan defaulted. Possible causes include
    events after origination, such as job loss or illness, that no application-time model
    can see.
* **Caveat:** PDP / ICE change one input while holding correlated inputs fixed (e.g. interest
  rate without sub-grade), so the extremes of some curves (very high incomes) are
  extrapolations into sparse regions.

### Optional extension: cost-sensitive lending (memo: `reports/memo_head_of_risk.md`)
* **Calibration:** CatBoost probabilities are already well calibrated (mean PD 15.2% vs 15.2%
  observed). An isotonic calibrator is still fitted for every model.
* **Empirical cost ratio from real cash flows (training loans):** a default loses $5,001
  on average (38% of principal, net of payments before charge-off and recoveries). A repaid
  loan earns $2,311 (16.8%). That puts the cost ratio at **≈ 2.2 : 1**, far below the 10 : 1
  used as an example in the brief.
* The optimal threshold on validation tracks the Bayes rule PD* = 1/(1+ratio). It falls from
  0.485 at 1:1 to 0.085 at 10:1 and 0.035 at 20:1, and the approval rate falls from 99.6% to
  32% to 10%.
* **Policies valued with each test loan's actual outcome:**

| Policy | Approval | Default rate of approved | Net profit (45k applicants) |
|---|---|---|---|
| Approve everyone (status quo) | 100% | 15.2% | $53.62M |
| Grades A-C only | 80% | 12.2% | $40.96M |
| Model, 10:1 cost-optimal (PD < 0.085) | 32% | 5.6% | $17.23M |
| Model, empirical 2.2:1 (PD < 0.30) | 92% | 13.2% | $51.21M |
| Model, best single cut-off (PD < 0.485) | 99.6% | 15.1% | $53.88M (+$0.26M) |
| **Model, loan-level expected value** | **97%** | **14.6%** | **$54.04M (+$0.42M)** |

  Lending Club's interest rates already price risk: the break-even PD rises from about 20%
  for grade A to about 60% for grade G (60-month loans). So a strict single PD cut-off, or a
  grade-only rule, throws away profitable high-yield loans. The best single cut-off only
  gains by declining almost nobody. The recommended rule approves a loan only when
  PD < gain / (gain + loss) for its grade and term. It gives the highest realised profit of
  all the policies.

---

## 4. User interface (`app/streamlit_app.py`)
`streamlit run app/streamlit_app.py`, integrated with the trained pipeline:
* **Score an applicant**: enter the loan, borrower and credit-file details. You get a live
  calibrated PD from all 9 models, a risk band, and an approve / decline decision at a chosen
  cost ratio. The page also shows a per-applicant explanation (SHAP or coefficient
  contributions) and a what-if curve for any input.
* **Batch scoring**: upload a CSV in the raw Lending Club format and download it with PDs and
  decisions added. `loan_amnt` and `annual_inc` are required. A missing `installment` is
  computed from amount, rate and term. Any other missing input is set to the typical training
  value, and the app lists which ones. An example file can be downloaded from the app.
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
│   ├── data_prep.py              # target, matured vintages, one loan per borrower, leakage, sampling, split
│   ├── features.py               # feature engineering, missing-value strategy, preprocessor
│   ├── eda.py                    # exploratory data analysis
│   ├── models.py                 # model factory, metrics, LoanModel wrapper
│   ├── train_benchmark.py        # training, tuning, benchmarking, PCA analysis
│   ├── borrower_check.py         # repeat borrowers left and their effect on test ROC-AUC
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
