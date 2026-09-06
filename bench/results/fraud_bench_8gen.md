| Scorer | PR-AUC | Precision @0.5% | Recall @0.5% |
|---|---:|---:|---:|
| evoml | 0.776 | 0.222 | 0.845 |
| logreg_balanced | 0.770 | 0.222 | 0.845 |
| random | 0.002 | 0.004 | 0.014 |

Champion: `net24x12(relu,lr=0.01,p=0.0)[29f]` · test rows 54114 · test frauds 71
AP difference 95% CI: vs random [0.698, 0.858] · vs logreg [-0.032, 0.052]
Gates: G1_beats_random=PASS, G2_recall_at_budget=PASS, G3_noninferior_to_logreg=PASS, beats_logreg_ci_excludes_zero=FAIL
