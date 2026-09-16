# Stats appendix

| Dataset | Model | n | Metric | Rerun mean | Rerun std (pop) | 95% CI (t) | Published | Published std | Delta | Delta / published std |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| FACED | cbramod | 5 | kappa | 0.4961 | 0.0124 | [0.4790, 0.5133] | 0.5041 | 0.0122 | -0.0080 | -0.7 |
| FACED | reve | 5 | kappa | 0.4190 | 0.0063 | [0.4103, 0.4277] | 0.5080 | 0.0191 | -0.0890 | -4.7 |
| HMC | cbramod | 5 | kappa | 0.6758 | 0.0026 | [0.6721, 0.6794] | nan | -- | +nan | +nan |
| HMC | reve | 5 | kappa | 0.6831 | 0.0046 | [0.6767, 0.6895] | 0.6982 | 0.0078 | -0.0151 | -1.9 |
| TUAB | cbramod | 5 | pr_auc | 0.8632 | 0.0056 | [0.8554, 0.8709] | 0.9221 | -- | -0.0589 | +nan |
| TUAB | reve | 5 | pr_auc | 0.8901 | 0.0114 | [0.8742, 0.9059] | 0.9281 | 0.0009 | -0.0380 | -42.3 |

Test-time BA per group: FACED / cbramod = 0.5551 +/- 0.0107; FACED / reve = 0.4850 +/- 0.0062; HMC / cbramod = 0.7279 +/- 0.0038; HMC / reve = 0.7337 +/- 0.0031; TUAB / cbramod = 0.7821 +/- 0.0099; TUAB / reve = 0.8052 +/- 0.0072

Blockers: published std is run-to-run variability, not the standard error of the comparison; no seed-level pairing exists against the published aggregate; groups with n=1 have no interval.
