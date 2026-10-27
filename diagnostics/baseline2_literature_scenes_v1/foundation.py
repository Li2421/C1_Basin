"""Opening provenance and historical inventory; no rollout or training."""
import ast
import csv
import io
import json
import re
import subprocess
from pathlib import Path
from audit import ROOT, OUT, FORBIDDEN, sha

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def main():
    assert (OUT / 'INTEGRITY_START.json').exists()
    external = {}
    for d in (OUT / 'external').iterdir():
        external[d.name] = {'commit': subprocess.check_output(['git','-C',str(d),'rev-parse','HEAD']).decode().strip(),
                            'files': {str(p.relative_to(d)): sha(p) for p in d.rglob('*') if p.is_file() and '.git' not in p.parts}}
    write('EXTERNAL_SOURCES.json', external)
    docs = list((ROOT / 'docs').glob('*.md'))
    candidates = list(docs)
    dirs = []
    for d in (ROOT / 'diagnostics').iterdir():
        if d == OUT or d.name.startswith(FORBIDDEN):
            continue
        if d.is_dir():
            dirs.append(d.name)
            candidates.extend(p for p in d.glob('*.md') if re.search('REPORT|HISTORY|HANDOFF|SCENE|PROVENANCE',p.name,re.I))
    matches = []
    pattern = re.compile(r'IMPC|2202\.06071|2207\.04915|2308\.10966|Jankovic|Chandra|crowded.random|circle.five|doorway|narrow.passage|门口|拥挤随机|圆形围场|窄通道', re.I)
    for p in candidates:
        for i, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
            if pattern.search(line):
                matches.append({'path':str(p.relative_to(ROOT)),'line':i,'text':line[:900]})
    dataset_inventory = []
    for d in (ROOT / 'datasets').iterdir():
        st = d.stat()
        dataset_inventory.append({'path':str(d.relative_to(ROOT)), 'size':st.st_size, 'mtime_ns':st.st_mtime_ns,
                                  'top_files': [p.name for p in d.iterdir()][:80] if d.is_dir() else []})
    write('HISTORY_SEARCH.json', {'diagnostic_directories':dirs,'searched_reports':[str(p.relative_to(ROOT)) for p in candidates],
                                'matches':matches,'datasets':dataset_inventory,'excluded_prefixes':FORBIDDEN})
    lines = ['# 历史证据检查', '', '先按目录名前缀剪枝，未打开、哈希或使用禁止前缀目录的文件。目录清单、实际检索文件、逐行命中和数据集目录收据见 HISTORY_SEARCH.json。',
             '', '既有窄通道审计 diagnostics/narrow_passage_part34_v1 是两机器人让行湾的历史轨迹审计，半径、目标、机器人数和执行语义与三机器人文献窄通道均不同，不能复用为本轮闭环标签。bottleneck_family 是文献启发的双向 Gap 母版，不是本轮四个参数实例。旧四路交叉口与对称对换也仅供机制参照。',
             '', '本轮保留跨类型六的控制器实现、冻结权重、候选动作、训练结构，以及跨场景物理特征定义作为可复用组件。critic 输入变化本身不否定旧标签；本轮标签缺口来自新的几何、初始状态、控制周期、机器人数与动力学。尚未完成逐身份共享数据库 preflight，不据文本检索宣称数据库不存在兼容记录。', '', '检索命中：', '']
    lines += [f"- {m['path']}:{m['line']}：{m['text']}" for m in matches if not Path(m['path']).name.startswith('prompt_')][:120]
    (OUT/'HISTORY_CHECK.md').write_text('\n'.join(lines)+'\n')
    rows = list(csv.DictReader((ROOT/'diagnostics/crosstype3_unseen_controller_schedule_v1/ZOO_MANIFEST.csv').open()))
    kinds = {'cross_entropy','distributed_cross_entropy','distributed_shooting','gradient_mpc','centralized_mppi','centralized_sampling','prioritized_astar','conflict_based_search'}
    selected = [r for r in rows if r['type_code'] in kinds]
    write('CONTROLLER_MANIFEST.json', selected)
    frozen = json.loads((ROOT/'diagnostics/crosstype6_shared_correction_controller_information_v1/FROZEN_PIPELINE.json').read_text())
    write('FROZEN_PIPELINE_INSPECTION.json', {'path':'diagnostics/crosstype6_shared_correction_controller_information_v1/FROZEN_PIPELINE.json',
          'sha256':sha(ROOT/'diagnostics/crosstype6_shared_correction_controller_information_v1/FROZEN_PIPELINE.json'),
          'top_keys':list(frozen),'outer_types':list(frozen.get('outer',{})), 'final_keys':list(frozen.get('final',{}))})
    # Extract exact GUI sampler methods without importing Qt or running GUI code.
    p=OUT/'external/IMPC-DR/2D/app.py'
    tree=ast.parse(p.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ControlPanel')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('set_random_ini','set_random_target')]
    text='import numpy as np\n\nclass OfficialSampler:\n'
    sources=[]
    for n in methods:
        source=ast.get_source_segment(p.read_text(),n)
        text+='\n'+'\n'.join('    '+s for s in source.splitlines())+'\n'
        sources.append({'method':n.name,'line':n.lineno,'end_line':n.end_lineno,'source_sha256':__import__('hashlib').sha256(source.encode()).hexdigest()})
    (OUT/'official_sampler.py').write_text(text)
    write('SAMPLER_PROVENANCE.json', {'source':str(p),'file_sha256':sha(p),'methods':sources,'change':'AST extraction and class wrapper only; method bodies unchanged'})
    print(json.dumps({'controllers':len(selected),'types':sorted(set(r['type_code'] for r in selected)), 'history_matches':len(matches),'external':{k:v['commit'] for k,v in external.items()}},ensure_ascii=False))

if __name__=='__main__':main()
