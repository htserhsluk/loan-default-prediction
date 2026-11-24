# Findings on the 2015 holdout
     
The test set contains 38,586 funded loans and 5,729 defaults (14.85%). The earlier training and validation cohorts have default rates of 12.80% and 13.69%. These rates apply to matched, resolved 36-month accounts. Differences in vintage coverage, origination mix, and resolution can all affect comparisons across years.

## B · Data and prediction visualization

The cohort plot shows how requested amounts overlap between repaid and defaulted loans. In the test cohort, 2,969 of 19,391 loans up to $10,000 defaulted (15.31%), compared with 819 of 6,088 loans above $20,000 (13.45%). Amount alone gives little separation. Months since the last public record is missing on 83.68% of sampled rows. Removing all rows with that missing field would discard most of the cohort. The fitted pipeline keeps a missingness indicator.

All ten precision–recall curves use identical test loans. Average precision ranges from 0.194 for the decision tree to 0.240 for logistic regression, above the prevalence reference of 0.148. The validation-selected logistic model has test ROC-AUC of 0.662. Its validation F1 threshold of 0.165 catches 2,646 of 5,729 defaults. Of 10,893 flagged loans, 8,247 repaid. The scores provide some ranking power, but classification errors remain frequent.

PCA retained 43 components explaining at least 95% of training variance. The paired logistic model's test average precision fell from 0.240 to 0.237 with PCA. Its separately selected threshold raised recall from 46.2% to 54.7% while lowering precision from 24.3% to 22.3%. That threshold change matters when interpreting the difference in confusion counts.

## C · Model efficiency

The benchmark records wall-clock fit time including preprocessing, batch scoring time per 1,000 loans, and serialized pipeline size for every model. Measurements are single runs on this machine. The random forest used about 13.2 MB and achieved average precision of 0.231. The selected logistic pipeline used 12.3 kB and achieved 0.240. Among the five prescribed boosters, XGBoost achieved the highest test average precision (0.235). The efficiency figure compares all configurations on log scales so that small pipelines remain visible.

## E · Error analysis

At their frozen validation thresholds, all ten models missed a larger fraction of actual defaults above $20,000 than at or below $10,000. For logistic regression, the missed-default fraction rose from 1,500/2,969 (50.5%) to 541/819 (66.1%). Meanwhile its false-alarm rate among repaid loans fell from 29.4% to 15.3%. The model therefore flagged fewer large loans relative to both outcome groups. The high-amount slice also has fewer observed defaults, so comparisons there carry greater sampling uncertainty. The data do not establish a causal effect of loan size.

The error table gives slice denominators and both error rates for every model. The benchmark table gives the corresponding full-cohort confusion counts. Thresholds reflect a validation F1 objective. A lending policy would need separate evidence on financial costs and future applicant populations.
