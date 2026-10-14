from core import *
from analyze import statistics
# Historical result regression; no new target-label reads.
old=ROOT/'diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8';z=np.load(old/'LABELS.npz');names=z['controller_names'].tolist();arms=z['arms'].tolist();selectors={c['controller']:c for c in read(old/'SELECTOR_MODELS.json')['choices']};got,_,_,_=statistics(z['success'],z['unknown'],names,arms,selectors);want=read(old/'results.json');checked=[]
for scheme in got['methods']:
 for k in ['mean_regret','max_regret','mean_gain_vs_F10','headroom_count','gain_vs_F10_CI95','gain_vs_source_average_CI95','mean_regret_unknown_bounds']:
  assert np.array_equal(np.asarray(got['methods'][scheme][k]),np.asarray(want['methods'][scheme][k])),(scheme,k);checked.append(scheme+':'+k)
for scheme in ['primary','secondary']:
 for k in ['Spearman','RMSE','source_mean_baseline_RMSE','RMSE_ratio']:
  assert got['prediction_metrics'][scheme][k]==want['prediction_metrics'][scheme][k],(scheme,k);checked.append(scheme+':'+k)
write(HERE/'STATISTICS_VERIFICATION.json',dict(utc=utc(),passed=True,historical_numeric_fields_exact=checked,only_cohort_fullness_changed_from8_to14=True,new_target_labels_read=False,source_results_sha256=sha(old/'results.json')))
print('Historical R12 numeric regression PASS')
