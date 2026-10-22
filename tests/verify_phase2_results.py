"""Validate full-size real FAST artifacts; never trains against held-out data."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from datasets import PHASE2_DATASETS,load_dataset,holdout_indices
from regression_pipeline import METHODS,feature_details,regression_metrics

ROOT=Path(__file__).resolve().parents[1]
counts={'datasets':0,'models':0,'training_predictions':0,'heldout_predictions':0}
with threadpool_limits(limits=1):
    for dataset_id,config in PHASE2_DATASETS.items():
        output=ROOT/'results'/dataset_id/'FAST'
        report=json.loads((output/'run.json').read_text(encoding='utf-8'))
        payload=json.loads((output/'dashboard_payload.json').read_text(encoding='utf-8'))
        assert payload==report
        assert report['configuration']['seed']==42
        assert report['configuration']['outer_folds']==2
        assert report['configuration']['inner_folds']==2
        assert report['runtime_seconds']>0
        assert report['original_input_count'] in (30,25)
        assert report['native_threads_per_worker']==1
        assert report['jobs']<=2
        assert report['cache_namespace']
        data=load_dataset(config)
        train,test=holdout_indices(data,random_state=42)
        np.testing.assert_array_equal(report['train_row_indices'],train)
        np.testing.assert_array_equal(report['test_row_indices'],test)
        assert not set(train)&set(test)
        assert set(m['method'] for m in report['methods'])==set(METHODS)
        assert report['best_model']['method']==min(report['methods'],key=lambda m:m['best_cv_rmse'])['method']
        train_predictions=pd.read_csv(output/'train_predictions.csv')
        test_predictions=pd.read_csv(output/'test_predictions.csv')
        np.testing.assert_array_equal(train_predictions.source_row_index,data.X.index[train])
        np.testing.assert_array_equal(test_predictions.source_row_index,data.X.index[test])
        if config.split_strategy=='chronological':
            assert len(train)==15788 and len(test)==3947
            dates=data.temporal_metadata.date
            assert dates.iloc[train[-1]]<dates.iloc[test[0]]
            assert pd.to_datetime(train_predictions.date).is_monotonic_increasing
            assert pd.to_datetime(test_predictions.date).is_monotonic_increasing
        coefficients=pd.read_csv(output/'coefficients.csv')
        table=pd.read_csv(output/'model_comparison.csv')
        assert len(table)==6
        assert list(table['Best CV RMSE'])==sorted(table['Best CV RMSE'])
        for method in report['methods']:
            name=method['method']
            model=joblib.load(output/'models'/(name.lower().replace(' ','_')+'.joblib'))
            assert isinstance(model['regressor'],LinearRegression)
            assert model.memory is None
            details=feature_details(model,config)
            assert details==method['features']
            assert len(details)==method['selected_encoded_count']
            assert 1<=len(details)<=method['encoded_input_count']
            assert len(set(d['original_name'] for d in details))==method['selected_original_count']
            if name=='Baseline':assert len(details)==method['encoded_input_count']
            num=model['preprocessor'].named_transformers_['numeric']
            np.testing.assert_allclose(num['imputer'].statistics_,data.X.iloc[train][list(config.numeric_columns)].median())
            # StandardScaler means follow median imputation, fitted on training only.
            imputed=num['imputer'].transform(data.X.iloc[train][list(config.numeric_columns)])
            np.testing.assert_allclose(num['scaler'].mean_,imputed.mean(axis=0))
            for indices,stored,part in [(train,train_predictions,'train'),(test,test_predictions,'test')]:
                predicted=model.predict(data.X.iloc[indices])
                assert np.isfinite(predicted).all()
                np.testing.assert_allclose(predicted,stored[name].to_numpy(),rtol=1e-10,atol=1e-7)
                metrics=regression_metrics(data.y.iloc[indices],predicted,len(details))
                for key,value in metrics.items():
                    if value is None:assert method[part+'_metrics'][key] is None
                    else:assert np.isclose(value,method[part+'_metrics'][key])
                counts['training_predictions' if part=='train' else 'heldout_predictions']+=len(predicted)
            selected_coefficients=coefficients[coefficients.method==name].sort_values('position')
            np.testing.assert_allclose(selected_coefficients.coefficient,[model['regressor'].intercept_,*model['regressor'].coef_])
            counts['models']+=1
        best=joblib.load(output/'best_model.joblib')
        np.testing.assert_allclose(best.predict(data.X.iloc[test]),test_predictions[report['best_model']['method']])
        outer=json.loads((output/'outer_folds.json').read_text(encoding='utf-8'))
        for record in outer['method_folds']:
            a,b=record['train_row_indices'],record['validation_row_indices']
            assert not set(a)&set(b)
            assert set(a)|set(b)<=set(train)
            assert not (set(a)|set(b))&set(test)
            if config.split_strategy=='chronological':assert max(a)<min(b)
        for workflow in outer['workflow_folds']:
            records=[r for r in outer['method_folds'] if r['outer_fold']==workflow['outer_fold']]
            selected=min(records,key=lambda r:r['inner_cv_rmse'])
            assert selected['method']==workflow['chosen_method']
        counts['datasets']+=1
# Both full-size experiments were rerun after cache-versioning changes.
first=json.loads((ROOT/'results/phase2_FAST_initial_summary.json').read_text(encoding='utf-8'))
current=json.loads((ROOT/'results/phase2_FAST_summary.json').read_text(encoding='utf-8'))
for a,b in zip(first['datasets'],current['datasets']):
    assert a['id']==b['id'] and a['best_method']==b['best_method']
    for m,n in zip(a['methods'],b['methods']):
        assert m['method']==n['method'] and m['selected_encoded_count']==n['selected_encoded_count']
        for key in ['best_cv_rmse','nested_cv_rmse','test_rmse','test_r2']:
            assert np.isclose(m[key],n[key],rtol=1e-10,atol=1e-7)
# Frontend bytes must be identical to the start of Phase 2.
before=json.loads((ROOT/'results/phase2_frontend_before.json').read_text(encoding='utf-8'))
after={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in (ROOT/'dashboard/dist').glob('*') if p.is_file()}
assert before==after,'Phase 2 modified the frontend'
verification={'status':'PASS',**counts,'repeated_fast_reproducible':True,'frontend_unchanged':True}
(ROOT/'results/phase2_validation.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print('PASS:',verification)
