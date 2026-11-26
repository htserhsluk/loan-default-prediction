# Memo - Cost-sensitive approval policy

**To:** Head of Risk
**From:** Credit Analytics Team (Kulshresth, Pranav Goyal, Aksha Alkesh Jain, Abhinav Bhatia, Jadav Sai Rugvedh)
**Re:** Choosing the loan approval threshold from the default-prediction model

---

## 1. Key messages
1. **The model ranks risk well.** CatBoost reaches a test ROC-AUC of 0.710.
   Its predicted probabilities of default (PD) are well calibrated: the average predicted PD is
   15.2% against an observed default rate of
   15.2%, and the Brier score is
   0.1191 (0.1192 after isotonic
   calibration). A PD threshold can therefore be read literally.
2. **Our own cash flows put the cost ratio near 2.2 : 1, not 10 : 1.**
   Across the training loans, a default loses on average $4,996
   (38% of principal, net of payments received before charge-off
   and recoveries). A repaid loan earns $2,283
   (16.7% of principal).
3. **A strict cut-off destroys value.** The 10 : 1 rule (PD < 9.0%) approves only
   34% of applicants and cuts realised profit from $52.58M to
   $17.86M.
4. **Risk is already priced.** Lending Club charges riskier grades higher rates, so the PD a
   loan can tolerate rises from 20% (grade A) to 60% (riskiest grades). A single
   PD cut-off for every loan ignores this. The best single cut-off on realised profit is very
   permissive (PD < 0.425, +$0.12M).
5. **Recommendation.** Adopt the **loan-level expected-value rule**: approve an applicant only when the calibrated PD is below the break-even PD of the loan's grade and term (between 20% and 60%). On the 45,000 held-out applicants it approves 96.8% of them, cuts the default rate of approved loans from 15.2% to 14.4%, and changes realised profit by **+$0.41M** vs approving everyone.

## 2. Policies compared (45,000 test applicants, realised cash flows)
| Policy | Approval rate | Default rate of approved | Defaulters declined | Good borrowers declined | Net profit | vs approve-all |
|---|---|---|---|---|---|---|
| Approve everyone (status quo) | 100.0% | 15.2% | 0% | 0% | $52.58M | +$0.00M |
| Grades A-B only | 53.9% | 9.0% | 68% | 42% | $25.66M | -$26.92M |
| Grades A-C only | 80.1% | 12.2% | 36% | 17% | $39.80M | -$12.78M |
| Grades A-D only | 93.1% | 13.9% | 15% | 5% | $47.32M | -$5.26M |
| Model: PD < 0.50 | 99.6% | 15.0% | 1% | 0% | $52.70M | +$0.12M |
| Model: F1-optimal (PD < 0.17) | 65.3% | 9.3% | 60% | 30% | $35.71M | -$16.86M |
| Model: 10:1 cost-optimal (PD < 0.090) | 33.7% | 5.7% | 87% | 63% | $17.86M | -$34.72M |
| Model: empirical 2.2:1 optimal (PD < 0.310) | 91.2% | 13.1% | 21% | 7% | $49.72M | -$2.86M |
| Model: profit-optimal single cut-off (PD < 0.425) | 99.6% | 15.0% | 1% | 0% | $52.70M | +$0.12M |
| Model: loan-level expected value (PD < break-even of grade & term) | 96.8% | 14.4% | 8% | 2% | $52.99M | +$0.41M |

* The most profitable policy on the test set is **Model: loan-level expected value (PD < break-even of grade & term)** ($52.99M).
* Grade-only rules are blunt. For example, approving grades A-C approves 80% and changes profit
  by -$12.78M, because it declines many good borrowers who pay high interest.
* Minimising *expected cost* with a flat ratio (2.2 : 1, PD < 31.0%)
  lowers the default rate to 13.1% but changes profit by -$2.86M.
  A flat ratio treats a 6%-interest and a 25%-interest loan as equally profitable.

## 3. How the optimal threshold moves with the cost ratio
| FN : FP cost | Optimal PD cut-off (validation) | Theory 1/(1+ratio) | Approval rate | Default rate of approved |
|---|---|---|---|---|
| 1 : 1 | 0.425 | 0.500 | 99.6% | 15.0% |
| 2 : 1 | 0.335 | 0.333 | 92.1% | 13.3% |
| 3 : 1 | 0.245 | 0.250 | 83.6% | 11.8% |
| 5 : 1 | 0.150 | 0.167 | 59.4% | 8.6% |
| 10 : 1 | 0.090 | 0.091 | 33.7% | 5.7% |
| 20 : 1 | 0.045 | 0.048 | 11.3% | 3.0% |

The empirical optimum tracks the Bayes rule PD* = 1/(1+ratio) closely. This confirms that the
calibrated PDs are reliable. As the assumed cost of a default rises, the cut-off falls quickly
and the approval rate collapses.

At 10 : 1, the lowest expected cost per applicant among all nine models comes from
**XGBoost** (0.7224), then CatBoost and
LightGBM. The worst is Decision Tree (0.7628).

## 4. Loan economics by grade and term (training loans)
| Grade | Term | Loans | Default rate | Gain if repaid | Loss if default | Implied ratio | Break-even PD |
|---|---|---|---|---|---|---|---|
| A | 36 | 44,734.0 | 5.5% | 9.4% | 37.7% | 4.0 : 1 | 19.9% |
| A | 60 | 328.0 | 6.7% | 12.8% | 42.6% | 3.3 : 1 | 23.1% |
| B | 36 | 65,418.0 | 11.4% | 14.2% | 36.9% | 2.6 : 1 | 27.8% |
| B | 60 | 2,471.0 | 15.2% | 23.7% | 40.7% | 1.7 : 1 | 36.8% |
| C | 36 | 50,162.0 | 18.4% | 18.0% | 38.3% | 2.1 : 1 | 32.0% |
| C | 60 | 5,995.0 | 21.5% | 31.9% | 33.8% | 1.1 : 1 | 48.5% |
| D | 36 | 22,930.0 | 23.7% | 22.4% | 39.0% | 1.7 : 1 | 36.5% |
| D | 60 | 4,047.0 | 26.7% | 37.1% | 34.6% | 0.9 : 1 | 51.7% |
| E | 36 | 6,669.0 | 30.4% | 25.7% | 41.2% | 1.6 : 1 | 38.4% |
| E | 60 | 3,350.0 | 31.9% | 42.0% | 31.2% | 0.7 : 1 | 57.3% |
| F | 36 | 1,434.0 | 35.6% | 31.2% | 41.1% | 1.3 : 1 | 43.2% |
| F | 60 | 1,789.0 | 35.9% | 45.1% | 33.4% | 0.7 : 1 | 57.4% |
| G | 36 | 239.0 | 41.0% | 32.3% | 48.4% | 1.5 : 1 | 40.0% |
| G | 60 | 434.0 | 37.8% | 48.4% | 32.0% | 0.7 : 1 | 60.2% |

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
