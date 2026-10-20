from common import *
import signal,subprocess
pids=subprocess.check_output(['pgrep','-f','/python diagnostics/blackbox_families_v1/advance_auto.py$'],text=True).split();assert len(pids)==1
os.kill(int(pids[0]),signal.SIGINT)
s=(HERE/'analysis.py').read_text();s=s.replace("names=f['controller_order'];states=", "names=[n for n in f['controller_order'] if not n.startswith('prioritized_astar')];states=")
s=s.replace("for c in read(HERE/'main_CASES.json'):\n  path=", "for c in read(HERE/'main_CASES.json'):\n  if c['family']=='prioritized_astar':continue\n  path=")
s=s.replace("for stage,cap in f['capabilities'].items():family_status.update(cap['family_status'])", "for stage,cap in f['capabilities'].items():family_status.update(cap['family_status'])\n observed_capability=dict(family_status);family_status['prioritized_astar']='GAP_IMPLEMENTATION'")
s=s.replace("reason='No eligible instance or source gate blocked'", "reason='Required oblique empty-map straightness unit test failed; initial axis-aligned test was insufficient. Observed pilot capability PASS 6/6, but family excluded from confirmatory evaluation.',observed_family_capability=observed_capability.get(family)")
s=s.replace("capability_pass_families=sum(family_status.get(k)=='PASS' for k in FAMILIES)", "capability_pass_families=sum(observed_capability.get(k)=='PASS' for k in FAMILIES),implementation_valid_families=sum(family_status.get(k)=='PASS' for k in FAMILIES)")
s=s.replace("row={'族':FULL_NAMES.get(family,'通过能力门的族合并'),'能力门':out.get('family_capability','合并')}","row={'族':FULL_NAMES.get(family,'能力门通过且实现合格的族合并'),'能力门':'六个实例通过；实现单元测试失败，确认性评估缺口' if out.get('family_capability')=='GAP_IMPLEMENTATION' else out.get('family_capability','合并')}")
(HERE/'analysis_valid_families.py').write_text(s)
write(HERE/'IMPLEMENTATION_GAP_ANALYSIS_AMENDMENT.json',dict(utc=utc(),reason='Preregistered handling of blocked family: required oblique empty-map A-star unit test failed. No evaluation outcome used to decide exclusion; all other families retain exact frozen selections and unchanged statistical functions. No controller parameter changed.',original_analysis_sha256=sha(HERE/'analysis.py'),new_analysis_sha256=sha(HERE/'analysis_valid_families.py'),statistics_sha256=sha(HERE/'analysis_statistics.py'),unit_test_sha256=sha(HERE/'ASTAR_OBLIQUE_UNIT_TEST.json'),excluded_family='prioritized_astar',prior_capability_gate_retained='PASS 6/6',evaluation_outcomes_read_to_choose_exclusion=False))
print('Only orchestration monitor stopped; all Slurm workers continue unchanged. Alternate blocked-family analysis written.')
