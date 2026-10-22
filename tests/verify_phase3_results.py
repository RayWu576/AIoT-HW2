"""Verify real saved outputs, all report figures and interval/metric provenance."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits
from datasets import PHASE2_DATASETS,load_dataset,make_cv
from regression_diagnostics import selected_matrix,design_information,rank_metrics
from regression_pipeline import METHODS

ROOT=Path(__file__).resolve().parents[1]
counts={'datasets':0,'models':0,'figures':0,'interval_rows':0,'metric_rows':0}
all_paths=set()
with threadpool_limits(limits=1):
    for key,config in PHASE2_DATASETS.items():
        source=ROOT/'results'/key/'FAST';tables=ROOT/'reports/tables'/key/'FAST'
        original=json.loads((source/'run.json').read_text())
        diagnostic=json.loads((tables/'diagnostics.json').read_text())
        assert diagnostic['dataset_id']==key and diagnostic['mode']=='FAST'
        assert diagnostic['configuration']==original['configuration']
        assert diagnostic['selected_method']==original['best_model']['method']
        assert diagnostic['source_run_sha256']==hashlib.sha256((source/'run.json').read_bytes()).hexdigest()
        assert set(diagnostic['models'])==set(METHODS)
        metrics=pd.read_csv(tables/'evaluation_metrics.csv')
        cv_table=pd.read_csv(tables/'cross_validation_metrics.csv')
        curves=pd.read_csv(tables/'training_cv_feature_count_curves.csv')
        assumptions=pd.read_csv(tables/'assumption_summary.csv')
        assert len(assumptions)==30
        assert len(metrics)==12 and len(cv_table)==6
        data=load_dataset(config);train=original['train_row_indices'];test=original['test_row_indices']
        for method in METHODS:
            slug=method.lower().replace(' ','_');path=source/'models'/(slug+'.joblib')
            model=joblib.load(path)
            assert diagnostic['models'][method]['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            info=design_information(selected_matrix(model,data.X.iloc[train]))
            assert info['rank']==diagnostic['models'][method]['rank']['rank']
            row=next(r for r in original['methods'] if r['method']==method)
            cvrow=cv_table[cv_table.method==method].iloc[0]
            assert np.isclose(cvrow.inner_cv_rmse,row['best_cv_rmse'])
            assert np.isclose(cvrow.outer_nested_cv_rmse,row['nested_cv_rmse'])
            for part,indices in [('train',train),('test',test)]:
                predicted=model.predict(data.X.iloc[indices]);actual=data.y.iloc[indices]
                found=metrics[(metrics.method==method)&(metrics.split==part)].iloc[0]
                expected=rank_metrics(actual,predicted,info['rank'])
                for name in ['mae','mse','rmse','r2','adjusted_r2']:
                    if expected[name] is None:assert pd.isna(found[name])
                    else:assert np.isclose(found[name],expected[name])
                for name in ['mae','mse','rmse','r2']:
                    assert np.isclose(found[name],row[part+'_metrics'][name])
                counts['metric_rows']+=1
            frame=pd.read_csv(tables/(slug+'_test_intervals.csv'))
            np.testing.assert_allclose(frame.prediction,model.predict(data.X.iloc[test]),rtol=1e-10,atol=1e-7)
            np.testing.assert_allclose(frame.actual,data.y.iloc[test])
            for lower,upper in [('mean_ci_lower','mean_ci_upper'),('individual_pi_lower','individual_pi_upper'),('robust_mean_ci_lower','robust_mean_ci_upper')]:
                valid=frame[lower].notna() & frame[upper].notna()
                assert valid.any()
                assert np.isfinite(frame.loc[valid,[lower,upper]]).all().all()
                assert (frame.loc[valid,lower]<=frame.loc[valid,upper]).all()
            valid=frame.individual_pi_lower.notna()
            assert (frame.loc[valid,'individual_pi_lower']<=frame.loc[valid,'mean_ci_lower']).all()
            assert (frame.loc[valid,'individual_pi_upper']>=frame.loc[valid,'mean_ci_upper']).all()
            assert frame.loc[~frame.estimable,'mean_ci_lower'].isna().all()
            if config.split_strategy=='chronological':assert pd.to_datetime(frame.date).is_monotonic_increasing
            assert len(diagnostic['models'][method]['assumptions'])==5
            counts['models']+=1;counts['interval_rows']+=len(frame)
        # Validate reconstruction against the actual stored fold scores and SST.
        splits=list(make_cv(config,original['configuration']['inner_folds'],random_state=42).split(data.X.iloc[train]))
        for method in METHODS:
            search=pd.read_csv(source/'cv_searches'/(method.lower().replace(' ','_')+'.csv'))
            if method=='LASSO':search=search.loc[[search.mean_test_score.idxmax()]]
            curve=curves[curves.method==method]
            assert len(curve)==len(search)
            for (_,saved),(_,point) in zip(search.iterrows(),curve.iterrows()):
                expected=[]
                for i,(_,val) in enumerate(splits):
                    y=data.y.iloc[np.asarray(train)[val]].to_numpy();sst=np.sum((y-y.mean())**2)
                    expected.append(1-len(val)*float(saved[f'split{i}_test_score'])**2/sst)
                assert np.isclose(point.validation_r2,np.mean(expected))
                assert np.isclose(point.cv_rmse,-saved.mean_test_score)
        assert len(diagnostic['figures'])==13
        for figure in diagnostic['figures']:
            path=ROOT/figure['path'];assert path.is_file()
            assert path not in all_paths;all_paths.add(path)
            assert key in figure['path'] and len(path.stem.split('_'))>=2
            with Image.open(path) as image:
                assert image.format=='PNG'
                assert min(image.size)>=1000
                assert min(image.info['dpi'])>=299
                image.verify()
            counts['figures']+=1
        counts['datasets']+=1
frontend=json.loads((ROOT/'results/phase2_frontend_before.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frontend.items())
result={'status':'PASS',**counts,'frontend_unchanged':True,'saved_predictions_unchanged':True}
(ROOT/'reports/phase3_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PASS:',result)
