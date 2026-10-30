"""Prespecified physical geometry variants and outcome-independent legality."""
from core import *
from scene_catalog import catalog
import copy
import datetime


def prepare():
    path = HERE/'CONSISTENT_BOX_SCENES.json'
    if path.exists() and read(path).get('complete'):
        return read(path)
    from physical_geometry import geometry
    scenes = catalog('A')
    audit = read(HERE/'STATIC_WALL_MARGIN_AUDIT.json')
    split = read(HERE/'SPLITS_FROZEN.json')['benchmark_A']
    summaries = []; variants = {}; counts = {}
    for item in audit['records']:
        if item['benchmark'] != 'A' or not item['mismatched_parameters']:
            continue
        name = item['scene']; scene = scenes[name]; category = scene['factor']
        difference = item['outer_centerline_threshold']-item['frozen_plant_centerline_threshold']
        if category == 'plus' and abs(difference)<.001:
            summaries.append(dict(scene=name, category=category, ignored_below_1mm=True, threshold_change=difference))
            continue
        revised = copy.deepcopy(scene)
        bc = copy.deepcopy(scene['bottleneck_config'])
        for key in ('agent_radius','wall_radius','wall_collision_margin','agent_collision_margin'):
            bc[key] = scene['config'][key]
        geom = geometry(bc)
        old_walls = np.asarray(scene['walls'], float).reshape(-1,2,2)
        assert geom.walls.shape == old_walls.shape and geom.walls.tobytes()==old_walls.tobytes()
        legal = []; state_rows = []
        for state in revised['states']:
            positions=np.asarray(state['positions']);goals=np.asarray(state['goals'])
            start_ok=bool(geom.valid_points(positions).all());goal_ok=bool(geom.valid_points(goals).all())
            ii,jj=np.triu_indices(state['N'],1)
            pair_ok=bool(np.all(np.linalg.norm(positions[ii]-positions[jj],axis=-1)>2*scene['config']['agent_radius']+scene['config']['agent_collision_margin']))
            valid=start_ok and goal_ok and pair_ok
            if valid:legal.append(state['index'])
            state['paper_geometry']=dict(config=bc,walls=geom.walls.tolist())
            state['paper_geometry_valid']=valid
            state_rows.append(dict(index=state['index'],start_wall_valid=start_ok,goal_wall_valid=goal_ok,initial_pair_valid=pair_ok,legal=valid))
        revised.update(original_definition_sha256=scene['definition_sha256'],
                       definition_sha256=digest(dict(original=scene['definition_sha256'],physical_geometry=bc)),
                       valid_state_indices=legal,physical_geometry_variant='consistent_v1')
        variants[name]=revised
        part=next(part for part in ('train','validation','test')if name in split[part])
        if part in ('validation','test'):
            selected=[r for r in state_rows if 8<=r['index']<=23]
            count=counts.setdefault(category,dict(legal=0,total=0));count['legal']+=sum(r['legal']for r in selected);count['total']+=len(selected)
        summaries.append(dict(scene=name,category=category,part=part,threshold_change=difference,
                              previous_wall_threshold=item['frozen_plant_centerline_threshold'],consistent_wall_threshold=geom.margin,
                              walls_bitwise_unchanged=True,states=state_rows))
    primary = {category:10*value['legal']>=9*value['total'] for category,value in counts.items()}
    output=dict(complete=True,time=datetime.datetime.now().astimezone().isoformat(),records=summaries,
                scene_records=variants,validation_test_legality=counts,primary_by_category=primary,
                criterion='At least90%legal validation+test states, separately for boxes and intersections; only legal states used in consistent-version evaluations.',
                selection_uses_rollout_outcomes=False,geometry='Literal wall centerlines unchanged; physical config gives wall threshold. Original visibility-roadmap constructor reused.',
                input_training_distribution='Frozen learned checkpoints trained on original inconsistent geometry; no retraining.',
                rollout_status='New-geometry learned validation/test replay follows original-environment reuse/acceleration certification; not yet evaluated by this static record.')
    dump(path,output)
    return output


def run():
    value=prepare()
    path=HERE/'scene_catalog.py';source=path.read_text()
    if 'version=None' not in source:
        source=source.replace('def catalog(benchmark):','def catalog(benchmark, version=None):')
        anchor="        result = {name: copy.deepcopy(all_scenes[name]) for name in names}"
        replacement=anchor+'''
        variant_path=HERE/'CONSISTENT_BOX_SCENES.json'
        if version!='original' and variant_path.exists():
            variants=read(variant_path)
            for name,scene in list(result.items()):
                if variants['primary_by_category'].get(scene['factor'],False) and name in variants['scene_records']:
                    result[name]=copy.deepcopy(variants['scene_records'][name])
'''
        assert anchor in source;source=source.replace(anchor,replacement)
        source=source.replace('def cohort(benchmark, part):','def cohort(benchmark, part, version=None):').replace('scenes = catalog(benchmark)','scenes = catalog(benchmark, version=version)')
        # Mandatory old-label certification remains on literal original physics.
        source=source.replace("    scenes = catalog('A')", "    scenes = catalog('A', version='original')")
        source+='''
def legal_indices(scene, indices):
    allowed=scene.get('valid_state_indices')
    return [i for i in indices if allowed is None or i in allowed]
'''
        path.write_text(source)
    # These modules have not performed any tuning or formal test yet.
    for filename in ('tuning.py','formal_tests.py'):
        target=HERE/filename;source=target.read_text()
        if 'import cohort,legal_indices' not in source:
            source=source.replace('from scene_catalog import cohort','from scene_catalog import cohort,legal_indices')
        if 'import catalog,legal_indices' not in source:
            source=source.replace('from scene_catalog import catalog','from scene_catalog import catalog,legal_indices')
        source=source.replace('for state in states:', 'for state in legal_indices(scene,states):')
        target.write_text(source)
    target=HERE/'certify_execution.py';source=target.read_text().replace("catalog('A')", "catalog('A',version='original')")
    source=source.replace("catalog(original_case['benchmark'])", "catalog(original_case['benchmark'],version='original')")
    target.write_text(source)
    target=HERE/'request_factory.py';source=target.read_text()
    start=source.index('def legacy_a_rng(')
    end=source.index('\n\n@lru_cache',start)
    block=source[start:end]
    block=block.replace("scene_definition_sha256=scene['definition_sha256'])", "scene_definition_sha256=scene.get('original_definition_sha256',scene['definition_sha256']))")
    source=source[:start]+block+source[end:]
    target.write_text(source)
    import importlib,scene_catalog,request_factory
    importlib.reload(scene_catalog);importlib.reload(request_factory)
    from freeze_execution import main as release
    release()
    with(HERE/'REPORT.md').open('a')as out:
        out.write('\n物理一致版本的主表选择（仅初态合法性，无rollout标签）：'+json.dumps(value['validation_test_legality'],ensure_ascii=False)+'；主表启用：'+json.dumps(value['primary_by_category'],ensure_ascii=False)+'。一致版本只纳入合法初态，冻结学习模型训练时使用旧版本，可能有分布变化。\n')
    return value
