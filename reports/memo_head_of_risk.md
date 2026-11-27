# Memo - Cost-sensitive approval policy

**To:** Head of Risk
**From:** Credit Analytics Team (Kulshresth, Pranav Goyal, Aksha Alkesh Jain, Abhinav Bhatia, Jadav Sai Rugvedh)
**Re:** Choosing the loan approval threshold from the default-prediction model

---

## 1. Key messages
1. **The model ranks risk well.** CatBoost reaches a test ROC-AUC of 0.709.
   Its predicted probabilities of default (PD) are well calibrated: the average predicted PD is
   15.2% against an observed default rate of
   15.2%, and the Brier score is
   0.1193 (0.1194 after isotonic
   calibration). A PD threshold can therefore be read literally.
2. **Our own cash flows put the cost ratio near 2.2 : 1, not 10 : 1.**
   Across the training loans, a default loses on average $5,001
   (38% of principal, net of payments received before charge-off
   and recoveries). A repaid loan earns $2,311
   (16.8% of principal).
3. **A strict cut-off destroys value.** The 10 : 1 rule (PD < 8.5%) approves only
   32% of applicants and cuts realised profit from $53.62M to
   $17.23M.
4. **Risk is already priced.** Lending Club charges riskier grades higher rates, so the PD a
   loan can tolerate rises from 20% (grade A) to 61% (riskiest grades). A single
   PD cut-off for every loan ignores this. The best single cut-off on realised profit is very
   permissive (PD < 0.485, +$0.26M).
5. **Recommendation.** Adopt the **loan-level expected-value rule**: approve an applicant only when the calibrated PD is below the break-even PD of the loan's grade and term (between 20% and 61%). On the 45,000 held-out applicants it approves 97.1% of them, cuts the default rate of approved loans from 15.2% to 14.6%, and changes realised profit by **+$0.42M** vs approving everyone.

## 2. Policies compared (45,000 test applicants, realised cash flows)
| Policy | Approval rate | Default rate of approved | Defaulters declined | Good borrowers declined | Net profit | vs approve-all |
|---|---|---|---|---|---|---|
| Approve everyone (status quo) | 100.0% | 15.2% | 0% | 0% | $53.62M | +$0.00M |
| Grades A-B only | 53.3% | 9.0% | 68% | 43% | $25.80M | -$27.82M |
| Grades A-C only | 80.3% | 12.2% | 36% | 17% | $40.96M | -$12.66M |
| Grades A-D only | 93.2% | 13.9% | 15% | 5% | $48.14M | -$5.48M |
| Model: PD < 0.50 | 99.6% | 15.1% | 1% | 0% | $53.88M | +$0.26M |
| Model: F1-optimal (PD < 0.17) | 65.2% | 9.3% | 60% | 30% | $36.35M | -$17.26M |
| Model: 10:1 cost-optimal (PD < 0.085) | 32.2% | 5.6% | 88% | 64% | $17.23M | -$36.39M |
| Model: empirical 2.2:1 optimal (PD < 0.300) | 91.6% | 13.2% | 21% | 6% | $51.21M | -$2.41M |
| Model: profit-optimal single cut-off (PD < 0.485) | 99.6% | 15.1% | 1% | 0% | $53.88M | +$0.26M |
| Model: loan-level expected value (PD < break-even of grade & term) | 97.1% | 14.6% | 7% | 2% | $54.04M | +$0.42M |

* The most profitable policy on the test set is **Model: loan-level expected value (PD < break-even of grade & term)** ($54.04M).
* Grade-only rules are blunt. For example, approving grades A-C approves 80% and changes profit
  by -$12.66M, because it declines many good borrowers who pay high interest.
* Minimising *expected cost* with a flat ratio (2.2 : 1, PD < 30.0%)
  lowers the default rate to 13.2% but changes profit by -$2.41M.
  A flat ratio treats a 6%-interest and a 25%-interest loan as equally profitable.

## 3. How the optimal threshold moves with the cost ratio
| FN : FP cost | Optimal PD cut-off (validation) | Theory 1/(1+ratio) | Approval rate | Default rate of approved |
|---|---|---|---|---|
| 1 : 1 | 0.485 | 0.500 | 99.6% | 15.1% |
| 2 : 1 | 0.330 | 0.333 | 92.9% | 13.4% |
| 3 : 1 | 0.250 | 0.250 | 87.4% | 12.5% |
| 5 : 1 | 0.160 | 0.167 | 65.2% | 9.3% |
| 10 : 1 | 0.085 | 0.091 | 32.2% | 5.6% |
| 20 : 1 | 0.035 | 0.048 | 10.2% | 2.7% |

The empirical optimum tracks the Bayes rule PD* = 1/(1+ratio) closely. This confirms that the
calibrated PDs are reliable. As the assumed cost of a default rises, the cut-off falls quickly
and the approval rate collapses.

At 10 : 1, the lowest expected cost per applicant among all nine models comes from
**CatBoost** (0.7211), then LightGBM and
XGBoost. The worst is Decision Tree (0.7559).

## 4. Loan economics by grade and term (training loans)
| Grade | Term | Loans | Default rate | Gain if repaid | Loss if default | Implied ratio | Break-even PD |
|---|---|---|---|---|---|---|---|
| A | 36 | 44,082 | 5.5% | 9.4% | 38.2% | 4.1 : 1 | 19.8% |
| A | 60 | 310 | 7.1% | 13.4% | 33.4% | 2.5 : 1 | 28.6% |
| B | 36 | 65,316 | 11.4% | 14.3% | 36.8% | 2.6 : 1 | 28.0% |
| B | 60 | 2,541 | 14.8% | 23.7% | 39.7% | 1.7 : 1 | 37.4% |
| C | 36 | 50,024 | 18.5% | 18.1% | 38.3% | 2.1 : 1 | 32.1% |
| C | 60 | 6,225 | 20.9% | 32.0% | 32.7% | 1.0 : 1 | 49.4% |
| D | 36 | 23,272 | 23.8% | 22.4% | 39.1% | 1.7 : 1 | 36.4% |
| D | 60 | 4,096 | 26.9% | 36.9% | 32.1% | 0.9 : 1 | 53.5% |
| E | 36 | 6,901 | 29.4% | 25.8% | 40.8% | 1.6 : 1 | 38.7% |
| E | 60 | 3,375 | 32.7% | 42.1% | 32.8% | 0.8 : 1 | 56.3% |
| F | 36 | 1,411 | 36.3% | 31.3% | 42.3% | 1.3 : 1 | 42.6% |
| F | 60 | 1,750 | 35.7% | 46.1% | 31.1% | 0.7 : 1 | 59.7% |
| G | 36 | 248 | 40.7% | 30.2% | 45.0% | 1.5 : 1 | 40.2% |
| G | 60 | 449 | 36.5% | 46.8% | 30.4% | 0.7 : 1 | 60.6% |

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
