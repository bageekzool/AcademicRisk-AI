# AcademicRisk AI - Model Comparison Report

Generated: 2026-11-12

Evaluation dataset: Validation split (664 students).
Test split has not been used in this report.

## 1. Multiclass Classification

Primary selection metric: Macro F1.

| model | stage | experiment | accuracy | macro_f1 | dropout_recall | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest A | A | cv_tuned | 0.5105 | 0.4745 | 0.4507 | 1.0040 |
| Random Forest A | A | baseline | 0.5648 | 0.4745 | 0.4225 | 1.0704 |
| Logistic Regression A | A | class_weight_custom | 0.5407 | 0.4711 | 0.5681 | 0.9718 |
| Logistic Regression A | A | cv_tuned | 0.5407 | 0.4702 | 0.5587 | 0.9767 |
| Logistic Regression A | A | class_weight_balanced | 0.5000 | 0.4669 | 0.4883 | 1.0082 |
| XGBoost A | A | baseline | 0.5723 | 0.4486 | 0.4695 | 0.9458 |
| Logistic Regression A | A | baseline | 0.5708 | 0.4363 | 0.4366 | 0.9377 |
| XGBoost A | A | cv_tuned | 0.5587 | 0.4314 | 0.4507 | 0.9473 |
| Logistic Regression B | B | class_weight_balanced | 0.6777 | 0.6346 | 0.6150 | 0.7746 |
| Logistic Regression B | B | class_weight_custom | 0.6973 | 0.6291 | 0.6714 | 0.7411 |
| Logistic Regression B | B | cv_tuned | 0.6717 | 0.6256 | 0.6056 | 0.7747 |
| Logistic Regression B | B | baseline | 0.7244 | 0.6156 | 0.7136 | 0.7142 |
| XGBoost B | B | baseline | 0.6973 | 0.6112 | 0.6338 | 0.7050 |
| XGBoost B | B | cv_tuned | 0.6973 | 0.6079 | 0.6197 | 0.6970 |
| Random Forest B | B | cv_tuned | 0.6521 | 0.5975 | 0.5681 | 0.7632 |
| Random Forest B | B | baseline | 0.6928 | 0.5869 | 0.6385 | 0.7242 |

### Best Macro F1 by Stage

| model | stage | experiment | macro_f1 | dropout_recall |
| --- | --- | --- | --- | --- |
| Random Forest A | A | cv_tuned | 0.4745 | 0.4507 |
| Logistic Regression B | B | class_weight_balanced | 0.6346 | 0.6150 |

## 2. Binary Classification

Risk ranking metric: Average Precision (AP).

| model | stage | accuracy | dropout_precision | dropout_recall | dropout_f1 | pr_auc_ap | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | A | 0.7093 | 0.5862 | 0.3192 | 0.4134 | 0.5301 | 0.6981 |
| xgboost | A | 0.6988 | 0.5489 | 0.3427 | 0.4220 | 0.5134 | 0.6849 |
| random_forest | A | 0.6657 | 0.4690 | 0.3192 | 0.3799 | 0.4667 | 0.6597 |
| xgboost | B | 0.7937 | 0.7346 | 0.5587 | 0.6347 | 0.7581 | 0.8437 |
| logistic | B | 0.8102 | 0.7736 | 0.5775 | 0.6613 | 0.7538 | 0.8459 |
| random_forest | B | 0.7922 | 0.7273 | 0.5634 | 0.6349 | 0.7365 | 0.8289 |

## 3. Threshold Optimization

Thresholds were selected using out-of-fold training predictions, targeting high F2 subject to OOF Precision >= 0.60.

| model | method | threshold | accuracy | precision | recall | f1 | f2 | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | default_0.50 | 0.5000 | 0.8102 | 0.7736 | 0.5775 | 0.6613 | 0.6083 | 123 | 36 | 90 | 415 |
| xgboost | default_0.50 | 0.5000 | 0.8012 | 0.7547 | 0.5634 | 0.6452 | 0.5935 | 120 | 39 | 93 | 412 |
| random_forest | default_0.50 | 0.5000 | 0.7907 | 0.7229 | 0.5634 | 0.6332 | 0.5894 | 120 | 46 | 93 | 405 |
| logistic | selected_oof | 0.2200 | 0.7605 | 0.5925 | 0.8122 | 0.6851 | 0.7561 | 173 | 119 | 40 | 332 |
| xgboost | selected_oof | 0.2100 | 0.7485 | 0.5816 | 0.7700 | 0.6626 | 0.7231 | 164 | 118 | 49 | 333 |
| random_forest | selected_oof | 0.2600 | 0.7395 | 0.5690 | 0.7746 | 0.6561 | 0.7224 | 165 | 125 | 48 | 326 |

### Selected Thresholds

| model | threshold | precision | recall | f2 | tp | fp |
| --- | --- | --- | --- | --- | --- | --- |
| logistic | 0.2200 | 0.5925 | 0.8122 | 0.7561 | 173 | 119 |
| xgboost | 0.2100 | 0.5816 | 0.7700 | 0.7231 | 164 | 118 |
| random_forest | 0.2600 | 0.5690 | 0.7746 | 0.7224 | 165 | 125 |

## 4. Probability Calibration

Lower Brier Score and Log Loss are better.

| model | brier_score | log_loss | average_precision | roc_auc |
| --- | --- | --- | --- | --- |
| raw | 0.1394 | 0.4440 | 0.7538 | 0.8459 |
| calibrated | 0.1391 | 0.4415 | 0.7524 | 0.8463 |

### Calibration Threshold Comparison

| model | method | threshold | precision | recall | f2 | tp | fp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | default_0.50 | 0.5000 | 0.7736 | 0.5775 | 0.6083 | 123 | 36 |
| raw | selected_oof | 0.2200 | 0.5925 | 0.8122 | 0.7561 | 173 | 119 |
| calibrated | default_0.50 | 0.5000 | 0.7785 | 0.5775 | 0.6089 | 123 | 35 |
| calibrated | selected_oof | 0.2300 | 0.5931 | 0.8075 | 0.7531 | 172 | 118 |

## 5. Model Selection Considerations

- Multiclass models are ranked primarily by validation Macro F1.
- Binary ranking quality is compared using Average Precision.
- Binary alert decisions are compared using F2, precision, recall, and alert counts.
- Precision >= 0.60 was an OOF threshold selection target, not a validation guarantee.
- Binary and multiclass accuracy values must not be treated as results of the same task.
- The Enrolled class remains challenging in the multiclass task.
- Model selection is provisional until the held-out test evaluation.

## 6. Evaluation Limitations

- The dataset represents historical academic outcomes and does not prove causality.
- Repeated model comparisons on validation can introduce selection bias.
- No prospective deployment evaluation has been performed.
- Alerts are intended for supportive review, not automated punitive decisions.
