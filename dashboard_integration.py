"""Phase 4 static export using verified Phase 2/3 artifacts and recorded candidates."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from threadpoolctl import threadpool_limits
from datasets import OFFICIAL_DATASETS,load_dataset
from datasets.descriptions import field_definition
from feature_selection import CachedRankingSelector
from regression_pipeline import METHODS,feature_details
from regression_diagnostics import (selected_matrix,design_information,rank_metrics,assumption_diagnostics,binned_residuals)
from phase3_evaluation import sanitize,cv_curve_table

ROOT=Path(__file__).resolve().parent


def preprocessing_descriptor(pre,config):
    numeric=pre.named_transformers_['numeric']
    result={'numericColumns':list(config.numeric_columns),'numericMedian':numeric['imputer'].statistics_.tolist(),
            'numericMean':numeric['scaler'].mean_.tolist(),'numericScale':numeric['scaler'].scale_.tolist(),
            'categoricalColumns':list(config.categorical_columns),'categoricalFill':[],'categories':[],
            'featureNames':pre.get_feature_names_out().tolist(),'manufacturerColumn':None,'manufacturerAliases':{}}
    if config.categorical_columns:
        cat=pre.named_transformers_['categorical']
        result['categoricalFill']=cat['imputer'].statistics_.tolist()
        result['categories']=[c.tolist() for c in cat['encoder'].categories_]
    return result


def interval_descriptor(Z,y,temporal):
    info=design_information(Z);B=info['basis'];ols=sm.OLS(y,B).fit()
    robust=None
    if temporal:robust=ols.get_robustcov_results(cov_type='HAC',maxlags=24,use_correction=True,use_t=False)
    elif np.max(1/len(y)+np.sum(B[:,1:]**2,axis=1))<1-1e-10:robust=ols.get_robustcov_results(cov_type='HC3',use_t=False)
    covariance=robust.cov_params() if robust is not None else None
    if covariance is not None and not np.isfinite(covariance).all():covariance=None
    descriptor={'center':info['center'].tolist(),'vectors':info['vectors'].tolist(),'scales':info['scales'].tolist(),
                'covariance':ols.cov_params().tolist(),'robustCovariance':None if covariance is None else covariance.tolist(),
                'sigma2':float(ols.mse_resid),'tCritical':float(stats.t.ppf(.975,info['df_resid'])),
                'zCritical':float(stats.norm.ppf(.975)),'rank':info['rank'],'residualDf':info['df_resid'],'maxTrainingLeverage':float(np.max(1/len(y)+np.sum(B[:,1:]**2,axis=1))),
                'robustType':'HAC (24 lags)' if temporal else 'HC3','estimabilityTolerance':1e-8,
                'label':'Approximate pointwise post-selection OLS intervals',
                'limitations':'Classical individual PI assumes iid homoscedastic normal errors. Robust mean CI is asymptotic; neither accounts for selection uncertainty or guarantees future coverage.'}
    return descriptor,info,ols


def display_indices(residual,limit=400):
    return np.unique(np.r_[np.linspace(0,len(residual)-1,min(limit,len(residual)),dtype=int),np.argmin(residual),np.argmax(residual)])


def export_dataset(config,mode='FAST'):
    source=ROOT/'results'/config.dataset_id/mode
    run=json.loads((source/'run.json').read_text(encoding='utf-8'))
    phase3=json.loads((ROOT/'reports/tables'/config.dataset_id/mode/'diagnostics.json').read_text(encoding='utf-8'))
    if phase3['source_run_sha256']!=hashlib.sha256((source/'run.json').read_bytes()).hexdigest():raise ValueError('Phase 3 diagnostics are stale')
    data=load_dataset(config)
    if run['source_sha256']!=hashlib.sha256(config.path.read_bytes()).hexdigest():raise ValueError('Source dataset changed')
    tr=np.asarray(run['train_row_indices']);te=np.asarray(run['test_row_indices'])
    X_train,X_test=data.X.iloc[tr],data.X.iloc[te];y_train,y_test=data.y.iloc[tr].to_numpy(),data.y.iloc[te].to_numpy()
    records={r['method']:r for r in run['methods']}
    saved={m:joblib.load(source/'models'/(m.lower().replace(' ','_')+'.joblib')) for m in METHODS}
    pre=saved['Baseline']['preprocessor'];descriptor=preprocessing_descriptor(pre,config)
    encoded_train=pre.transform(X_train);inputs=[]
    for name in config.feature_names:
        definition=field_definition(config,name)
        if name in config.numeric_columns:
            i=list(config.numeric_columns).index(name);col=X_train[name]
            inputs.append({'name':name,**definition,'kind':'numeric','default':float(descriptor['numericMedian'][i]),
                           'min':float(col.min()),'max':float(col.max()),'step':1 if definition['integer'] else .01})
        else:
            i=list(config.categorical_columns).index(name)
            inputs.append({'name':name,**definition,'kind':'categorical','default':str(descriptor['categoricalFill'][i]),'options':descriptor['categories'][i]})
    curves=cv_curve_table(source,config,data,run)
    exported=[];comparison=[];private=ROOT/'results/dashboard'/mode/config.dataset_id/'models';private.mkdir(parents=True,exist_ok=True)
    temporal=config.split_strategy=='chronological';ordered=np.argsort(X_train.index)
    for method in METHODS:
        frame=pd.read_csv(source/'cv_searches'/(method.lower().replace(' ','_')+'.csv'))
        method_curves=curves[curves.method==method]
        candidates=[]
        for number,row in frame.iterrows():
            params=ast.literal_eval(row['params']);official=params==records[method]['best_params']
            fitted=saved[method] if official else copy.deepcopy(saved[method])
            if not official:
                if method=='LASSO':fitted.set_params(ranking=CachedRankingSelector('LASSO',alpha_fraction=params['ranking__alpha_fraction'],random_state=run['configuration']['seed']).fit(encoded_train,y_train))
                if method!='Baseline':
                    ranked=fitted['ranking'].transform(encoded_train)
                    fitted['selector'].set_params(k=params.get('selector__k','all')).fit(ranked,y_train)
                    fitted['regressor'].fit(fitted['selector'].transform(ranked),y_train)
            details=feature_details(fitted,config);k=len(details)
            candidates.append((number,row,params,official,fitted,details,k))
        # When alpha candidates have the same final k, expose the training-CV best.
        by_count={}
        for item in candidates:
            k=item[-1]
            if k not in by_count or item[1].mean_test_score>by_count[k][1].mean_test_score:by_count[k]=item
        for number,row,params,official,fitted,details,k in sorted(by_count.values(),key=lambda c:c[-1]):
            identifier=method.lower().replace(' ','_')+'-k'+str(k)+'-c'+str(number)
            names=[d['encoded_name'] for d in details];Z=selected_matrix(fitted,X_train);Z_test=selected_matrix(fitted,X_test)
            ptrain=fitted.predict(X_train);ptest=fitted.predict(X_test);residual=y_train-ptrain
            uncertainty,info,ols=interval_descriptor(Z,y_train,temporal)
            if official:
                assumptions=phase3['models'][method]['assumptions']
                vif=pd.read_csv(ROOT/'reports/tables'/config.dataset_id/mode/(method.lower().replace(' ','_')+'_vif.csv'))
            else:assumptions,vif,_,_=assumption_diagnostics(Z[ordered],y_train[ordered],ptrain[ordered],names,temporal=temporal)
            A=np.column_stack([np.ones(len(Z)),Z]);mapping=np.linalg.pinv(A)@info['basis']
            covariance=mapping@ols.cov_params()@mapping.T;se=np.sqrt(np.maximum(0,np.diag(covariance)))
            coefficients=[];values=np.r_[fitted['regressor'].intercept_,fitted['regressor'].coef_]
            for i,(name,value) in enumerate(zip(['Intercept',*names],values)):
                identified=bool(info['coefficient_estimable'][i]);t=float(value/se[i]) if identified and se[i]>0 else None
                width=uncertainty['tCritical']*se[i] if identified else None
                coefficients.append({'name':name,'estimate':float(value),'identifiable':identified,
                                     'se':float(se[i]) if identified else None,'t':t,
                                     'p':float(2*stats.t.sf(abs(t),info['df_resid'])) if t is not None else None,
                                     'low':float(value-width) if width is not None else None,'high':float(value+width) if width is not None else None,
                                     'vif':None if i==0 else vif.iloc[i-1]['vif'],
                                     'vifStatus':'intercept' if i==0 else vif.iloc[i-1]['status']})
            sample=display_indices(residual);osm,fitline=stats.probplot(residual,dist='norm');qq_idx=np.linspace(0,len(residual)-1,min(201,len(residual)),dtype=int)
            edges=np.histogram_bin_edges(np.r_[residual,y_test-ptest],bins=45)
            hist=np.histogram(residual,bins=edges)[0];testhist=np.histogram(y_test-ptest,bins=edges)[0]
            group=binned_residuals(ptrain,residual)
            if method=='LASSO':
                # Exact fold R2 for each alpha reconstructed from recorded RMSE.
                r2row=cv_curve_table_for_candidate(row,config,data,run)
            else:r2row=method_curves.iloc[number].to_dict()
            model={'id':identifier,'method':method,'k':k,'isOfficial':official,'params':params,
                   'features':names,'originalFeatures':list(dict.fromkeys(d['original_name'] for d in details)),
                   'intercept':float(values[0]),'weights':values[1:].tolist(),'coefficients':coefficients,
                   'train':rank_metrics(y_train,ptrain,info['rank']),'test':rank_metrics(y_test,ptest,info['rank']),
                   'cvRmse':float(-row.mean_test_score),'cvStd':float(row.std_test_score),'validationR2':r2row['validation_r2'],
                   'nestedCvRmse':records[method]['nested_cv_rmse'] if official else None,'nestedCvStd':records[method]['nested_cv_std'] if official else None,
                   'rank':info['rank'],'f':float(ols.fvalue),'fP':float(ols.f_pvalue),
                   'testPredictions':ptest.tolist(),'uncertainty':uncertainty,'assumptions':assumptions,
                   'diagnostics':{'residualScatter':np.column_stack([ptrain[sample],residual[sample]]).tolist(),
                                  'displayedTrainingPoints':len(sample),'trainingSize':len(Z),
                                  'binnedResiduals':group.to_dict(orient='records'),
                                  'histogram':{'edges':edges.tolist(),'train':hist.tolist(),'test':testhist.tolist(),'trainDensity':(hist/(len(residual)*np.diff(edges))).tolist(),'testDensity':(testhist/(len(y_test)*np.diff(edges))).tolist()},
                                  'qq':np.column_stack([osm[0][qq_idx],osm[1][qq_idx]]).tolist(),
                                  'qqLine':{'slope':float(fitline[0]),'intercept':float(fitline[1])},
                                  'acf':assumptions[1]['statistic']['acf']}}
            exported.append(model);joblib.dump(fitted,private/(identifier+'.joblib'),compress=3)
            if official:comparison.append({'method':method,'modelId':identifier,'k':k,'rank':info['rank'],'cvRmse':model['cvRmse'],'cvStd':model['cvStd'],
                                           'nestedCvRmse':model['nestedCvRmse'],'nestedCvStd':model['nestedCvStd'],'train':model['train'],'test':model['test']})
        print(config.dataset_id,method,'exported k:',sorted(by_count),flush=True)
    recommended=next(m for m in exported if m['method']==run['best_model']['method'] and m['isOfficial'])
    comparison.sort(key=lambda r:r['cvRmse'])
    rows=[{'id':int(i)+1,'actual':float(y)} for i,y in zip(X_test.index,y_test)]
    if temporal:
        for r,date in zip(rows,data.temporal_metadata.date.iloc[te]):r['date']=str(date)
    metadata={'id':config.dataset_id,'title':'Student Performance Prediction' if not temporal else 'Appliances Energy Prediction',
              'problem':('Problem 1: Student Performance Prediction' if not temporal else 'Problem 2: Appliances Energy Prediction'),
              'description':'Early prediction of Mathematics final grade from background and study characteristics.' if not temporal else 'Predict appliance energy use from contemporaneous household sensors and weather.',
              'target':config.target,'unit':config.target_unit,'originalFeatureCount':config.original_feature_count,
              'encodedFeatureCount':len(descriptor['featureNames']),'rawSize':len(data.y),'sampleLabel':'Student' if not temporal else 'Observation',
              'splitDescription':'Seed=42 shuffled 80/20; nested KFold' if not temporal else 'First 80% train / last 20% test; expanding TimeSeriesSplit at both CV levels',
              'source':'https://archive.ics.uci.edu/dataset/320/student+performance' if not temporal else 'https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction',
              'kaggleSource':'https://www.kaggle.com/datasets/larsen0966/student-performance-data-set' if not temporal else 'https://www.kaggle.com/datasets/loveall/appliances-energy-prediction',
              'sourceNote':'Requested Kaggle archive supplies Portuguese data only; authentic Mathematics file comes from UCI (395 rows).' if not temporal else 'Authentic Kaggle copy; date retained only as metadata, rv1/rv2 excluded; lights is retained.',
              'targetNote':'G1/G2 are historical grades and excluded from the official early-30 scenario.' if not temporal else 'Chronological residuals are dependent; intervals are working-model approximations, not forecast guarantees.',
              'fieldLabels':{f['name']:f['label'] for f in inputs},'timeRanges':run.get('time_ranges')}
    return sanitize({'metadata':metadata,'mode':mode,'configuration':run['configuration'],'generatedAt':run['generated_at_utc'],
                     'trainSize':len(tr),'testSize':len(te),'preprocessing':descriptor,'inputs':inputs,'models':exported,'comparison':comparison,
                     'testRows':rows,'defaultModelId':recommended['id'],'defaultMethod':recommended['method'],'defaultK':recommended['k'],
                     'sourceRunSha256':hashlib.sha256((source/'run.json').read_bytes()).hexdigest(),
                     'ranges':{name:{'min':float(encoded_train[:,i].min()),'max':float(encoded_train[:,i].max())} for i,name in enumerate(descriptor['featureNames'])}})


def cv_curve_table_for_candidate(row,config,data,run):
    from datasets import make_cv
    tr=np.asarray(run['train_row_indices']);scores=[]
    for i,(_,validation) in enumerate(make_cv(config,run['configuration']['inner_folds'],gap=run['configuration']['gap'],random_state=run['configuration']['seed']).split(tr)):
        y=data.y.iloc[tr[validation]].to_numpy();sst=np.sum((y-y.mean())**2)
        scores.append(1-len(y)*float(row[f'split{i}_test_score'])**2/sst if sst>0 else np.nan)
    return {'validation_r2':float(np.nanmean(scores))}


def export_dashboard(mode='FAST'):
    payloads={}
    with threadpool_limits(limits=1):
        for key,config in OFFICIAL_DATASETS.items():payloads[key]=export_dataset(config,mode)
    bundle={'schemaVersion':5,'mode':mode,'repository':'https://github.com/RayWu576/AIoT-HW2','datasets':payloads}
    out=ROOT/'dashboard/dist';encoded=json.dumps(bundle,ensure_ascii=True,allow_nan=False,separators=(',',':'))
    (out/'data.json').write_text(encoded,encoding='utf-8');(out/'data.js').write_text('window.DASHBOARD_DATA = '+encoded+';\n',encoding='utf-8')
    from dashboard_export import refresh_asset_versions
    refresh_asset_versions(out)
    print('Phase 4 static bundle bytes:',len(encoded),'official IDs:',list(payloads),flush=True)
    return bundle
