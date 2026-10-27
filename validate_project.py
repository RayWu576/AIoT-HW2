"""Repeatable Phase 5 FAST end-to-end QA; never commits, pushes or deploys."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/phase5'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def run_command(name,args):
    started=perf_counter()
    command=[sys.executable,*args]
    print('RUN '+name+': '+' '.join(args),flush=True)
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',
                          errors='replace',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    elapsed=perf_counter()-started
    text=result.stdout+'\n'+result.stderr
    (OUT/(name+'.log')).write_text(text,encoding='utf-8')
    record={'name':name,'command':command,'returncode':result.returncode,
            'runtime_seconds':elapsed,'log':'reports/phase5/'+name+'.log'}
    if name=='unit_tests':
        match=re.search(r'Ran (\d+) tests?',text)
        record['tests_run']=int(match.group(1)) if match else 0
        for key in ['failures','errors','skipped']:
            match=re.search(key+r'=(\d+)',text)
            record[key]=int(match.group(1)) if match else 0
        record['passed']=record['tests_run']-record['failures']-record['errors']-record['skipped']
    print(name+': '+('PASS' if result.returncode==0 else 'FAIL')+f' ({elapsed:.2f}s)',flush=True)
    if result.returncode:
        print(text[-4000:],flush=True)
    return record

def audit():
    excluded={'.git','.cache','.venv','venv','__pycache__'}
    files=[p for p in ROOT.rglob('*') if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts)]
    python=[p for p in files if p.suffix=='.py']
    for p in python:
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    bundle=json.loads((ROOT/'dashboard/dist/data.json').read_text(encoding='utf-8'))
    assert set(bundle['datasets'])=={'student_performance','appliances_energy'}
    assert bundle['schemaVersion']==5
    for key,count in [('student_performance',30),('appliances_energy',25)]:
        assert bundle['datasets'][key]['metadata']['originalFeatureCount']==count
    figures=list((ROOT/'reports/figures').rglob('*.png'))
    assert len(figures)==26
    inventory={'file_count':len(files),'python_syntax_files':len(python),'official_figures':len(figures),
               'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size} for p in files]}
    (OUT/'audit_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    packages={n:importlib.metadata.version(n) for n in
              ['numpy','pandas','matplotlib','joblib','scikit-learn','statsmodels','scipy','threadpoolctl','Pillow','playwright']}
    environment={'python':platform.python_version(),'platform':platform.platform(),'packages':packages}
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2)+'\n',encoding='utf-8')
    # Compile-time audit plus safe imports of all current root/data modules.
    modules=[p.stem for p in ROOT.glob('*.py') if p.stem!='validate_project']
    import_code='import importlib; '+ '; '.join('importlib.import_module('+repr(m)+')' for m in modules)
    imported=run_command('module_import_audit',['-c',import_code])
    if imported['returncode']:
        raise RuntimeError('Module import audit failed; see log')
    return {'files_inventoried':len(files),'python_syntax_files':len(python),
            'root_modules_imported':len(modules),'figures':len(figures)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-training',action='store_true',
                        help='Reuse an explicitly completed real FAST run; still regenerate downstream artifacts')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    protected=[p for p in (ROOT/'datasets').rglob('*.csv')]
    protected+=list((ROOT/'historical').rglob('*')) if (ROOT/'historical').is_dir() else []
    before={p.relative_to(ROOT).as_posix():sha(p) for p in protected if p.is_file()}
    steps=[]
    commands=[]
    if not args.reuse_training:
        commands.append(('training',['phase2_experiment.py','--mode','FAST','--jobs','2']))
    commands += [
        ('diagnostics',['phase3_evaluation.py','--mode','FAST']),
        ('dashboard_export',['dashboard_export.py','--mode','FAST']),
        ('report_preparation',['report_preparation.py','--mode','FAST']),
        ('unit_tests',['-m','unittest','discover','-s','tests','-p','test_*.py','-v']),
        ('backend_integration',['tests/verify_phase2_results.py']),
        ('diagnostic_integration',['tests/verify_phase3_results.py']),
        ('browser_integration',['tests/verify_official_dashboard.py']),
    ]
    for name,command in commands:
        record=run_command(name,command);steps.append(record)
        if record['returncode']:
            result={'passed':False,'failed_step':name,'steps':steps,'runtime_seconds':perf_counter()-started}
            (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
            return 1
    inspected=audit()
    assert before=={name:sha(ROOT/name) for name in before},'Raw/historical files unexpectedly changed'
    runs=[json.loads((ROOT/'results'/key/'FAST/run.json').read_text(encoding='utf-8'))
          for key in ['student_performance','appliances_energy']]
    result={'passed':True,'mode':'FAST','full_executed':False,
            'training_reused':args.reuse_training,'steps':steps,
            'audit':inspected,'protected_files_unchanged':len(before),
            'datasets':[{'id':r['dataset_id'],'pipeline_seconds':r['runtime_seconds'],
                         'cache_namespace_preexisting':r['cache_namespace_preexisting'],
                         'run_sha256':sha(ROOT/'results'/r['dataset_id']/'FAST/run.json'),
                         'selected_method':r['best_model']['method']} for r in runs],
            'unit_tests':next(s for s in steps if s['name']=='unit_tests'),
            'backend':json.loads((ROOT/'results/phase2_validation.json').read_text(encoding='utf-8')),
            'diagnostics':json.loads((ROOT/'reports/phase3_validation.json').read_text(encoding='utf-8')),
            'browser':json.loads((ROOT/'reports/phase4_validation.json').read_text(encoding='utf-8')),
            'runtime_seconds':perf_counter()-started,
            'deployment_verified':False,'commit_or_push_performed':False,
            'manual_remaining':['NotebookLM source-grounded >=100 Chinese-character summary',
                                'Actual ChatGPT conversation PDF','Final report PDF export']}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['browser','steps']},indent=2),flush=True)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
