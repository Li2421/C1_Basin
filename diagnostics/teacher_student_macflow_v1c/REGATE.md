# Revised distillation gate (frozen checkpoints only)

| Student | Previously valid | Step | All cosine | Cosine ≥.05/.10/.20 | ≤.05m/s | New decision |
|---|---|---:|---:|---|---:|---|
| base_u_plus_w_v4_wide_macflow | False | 45500 | 0.76166 | 0.98933/0.99042/0.99361 | 94.78% | True |
| base_u_macflow | True | 18500 | 0.95468 | 0.99870/0.99952/0.99955 | 97.31% | unchanged |
| base_u_v4_macflow | False | 76000 | 0.85771 | 0.99568/0.99601/0.99701 | 97.03% | True |
| base_u_plus_w_v6_macflow | False | 77000 | 0.85327 | 0.99647/0.99789/0.99897 | 97.51% | True |
| base_u_plus_w_v3_macflow | False | 4500 | 0.88048 | 0.97935/0.98673/0.99014 | 89.90% | True |
| base_u_plus_w_v4_train_targeted_macflow | False | 68500 | 0.84320 | 0.99510/0.99690/0.99854 | 96.81% | True |
| base_u_v3_macflow | False | 4000 | 0.85952 | 0.97930/0.98409/0.98737 | 90.66% | True |
| base_u_v2_macflow | True | 47000 | 0.92881 | 0.99882/0.99932/0.99949 | 97.60% | unchanged |
| base_u_v10_local_macflow | True | 75000 | 0.98823 | 0.99941/0.99948/0.99957 | 98.78% | unchanged |
| base_u_plus_w_v5_macflow | False | 71500 | 0.85024 | 0.99739/0.99883/0.99901 | 98.34% | True |
| base_u_v7_local_macflow | True | 77000 | 0.98304 | 0.99886/0.99917/0.99939 | 98.22% | unchanged |
| base_u_plus_w_v4_macflow | False | 78000 | 0.80546 | 0.99198/0.99367/0.99503 | 94.62% | True |
| base_u_v8_local_macflow | True | 71500 | 0.98513 | 0.99879/0.99930/0.99932 | 98.30% | unchanged |
| base_u_v9_local_macflow | True | 67000 | 0.98269 | 0.99899/0.99932/0.99941 | 98.55% | unchanged |
