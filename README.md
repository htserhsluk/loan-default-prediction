# Loan default prediction

This project estimates default risk for historical, funded LendingClub loans with a 36-month term. It uses the supplied Kaggle release for predictors and resolved outcomes, and an older LendingClub release to select distinct borrower members. The interactive demo uses the same fitted pipeline as the benchmark.

## Reproduce

Run these PowerShell commands from the checkout with Python 3.13. The source CSV in this workspace is one level above the checkout inside the loan.csv folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python archive.py
python prepare.py ../loan.csv/loan.csv
python benchmark.py
python analyze.py
python -m pytest -q
python -m streamlit run app.py
```

Download and unpack the primary CSV before running preparation. The archive command downloads three historical vintages and the lender dictionary from a pinned university archive. The compressed 100,000-row extract is written to `data/sample.csv.gz`. Local fitted models, row-level predictions, borrower verification inputs, and source files are excluded from Git. [Primary CSV](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) · [Historical files and lender dictionary](https://github.com/ChicagoBoothML/DATA___LendingClub/tree/66a86bb5ddbe6d698fbe4b0180b1475b591a632c)

## Study design

- **Borrower unit.** The supplied file has blank borrower IDs. The historical lender dictionary defines `member_id` as the borrower member identifier. Start with archive members appearing exactly once across all three historical files. Align their records with the newer release using an exact signature of issue month, term, amount, instalment, rate, subgrade, purpose, and four credit-file counts. Require complete fields and a unique signature in each release. The final extract retains only a file row number for audit. The method depends on origination fields being stable across releases. Missing, modified or ambiguous records can change the composition of the matched cohort.
- **Target and sampling.** Restrict to loans issued in 2012–2015 with a scheduled 36-month term and a resolved label. Fully Paid is 0. Charged Off and Default are 1. Draw exactly 100,000 eligible matched records uniformly without replacement (seed 42). This limits computation and preserves the observed event mix in expectation. Estimates apply to funded, matched, resolved loans in these vintages. Rejected applicants, unresolved accounts and longer terms fall outside the study.
- **Predictors.** The explicit allowlist in `prepare.py` covers the request amount, application details and credit history as recorded at origination. Historical credit-file delinquencies and collections precede this loan. Later recoveries, payments, balance updates, settlement and hardship fields can reveal its outcome and are excluded. Model inputs also exclude IDs, free text, issue year and lender-assigned pricing and grades. Matching fields have a separate role in data preparation.
- **Missingness.** Train-only median imputation with missingness indicators handles numeric fields, including months since past adverse events. A missing event date may mean no recorded event or unavailable history. Categorical inputs use a missing category and allow unseen levels. All transformations, including PCA, fit on training rows only.
- **Evaluation.** Train on 2012–2013, tune thresholds and select the model on 2014, and report held-out results on 2015. Each model's threshold maximizes validation F1. The selected model has the highest validation average precision. This retrospective issue-year comparison uses eventual resolved outcomes. A historical deployment simulation would additionally require the date each training label became observable.

## Outputs

| Phase | Evidence |
| --- | --- |
| Data preparation | [Source audit](reports/audit.json), [year-split counts](reports/cohorts.csv), [missingness and feature variation](reports/features.csv), [cohort figure](reports/cohort.png) |
| Benchmark | [Validation and test metrics, thresholds and confusion counts](reports/benchmark.csv) |
| Three analyses | [Findings](reports/findings.md), [precision–recall](reports/precision-recall.png), [efficiency](reports/efficiency.png), [error rates by amount](reports/errors.png), [underlying denominators](reports/errors.csv) |
| Interactive demo | Run the Streamlit command above, edit the example loan, choose **Score application**, then open **Benchmark results**. The displayed percentage is a model score. Calibration on future applicants has not been established. |
