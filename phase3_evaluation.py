"""Phase 3: evaluate saved Phase 2 models; no training search or frontend writes."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from threadpoolctl import threadpool_limits
from datasets import PHASE2_DATASETS, load_dataset, make_cv
from regression_pipeline import METHODS, feature_details
from regression_diagnostics import (selected_matrix, design_information, rank_metrics,
                                    assumption_diagnostics, prediction_intervals)

ROOT=Path(__file__).resolve().parent
COLORS=['#64748b','#2563eb','#059669','#d97706','#9333ea','#dc2626']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':100,'savefig.dpi':300})


def sanitize(value):
    if isinstance(value,np.generic):return sanitize(value.item())
    if isinstance(value,np.ndarray):return sanitize(value.tolist())
    if isinstance(value,dict):return {k:sanitize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [sanitize(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    path.write_text(json.dumps(sanitize(value),indent=2,allow_nan=False),encoding='utf-8')


def cv_curve_table(source,config,data,run):
    train=np.asarray(run['train_row_indices'])
    cv=make_cv(config,run['configuration']['inner_folds'],gap=run['configuration']['gap'],random_state=run['configuration']['seed'])
    splits=list(cv.split(data.X.iloc[train]))
    metadata={r['method']:r for r in run['methods']};rows=[]
    for method in METHODS:
        frame=pd.read_csv(source/'cv_searches'/(method.lower().replace(' ','_')+'.csv'))
        if method=='LASSO':frame=frame.loc[[frame.mean_test_score.idxmax()]]
        for _,candidate in frame.iterrows():
            requested=candidate.get('param_selector__k','all')
            if method=='LASSO':
                count=metadata[method]['selected_encoded_count']; semantics='Best alpha only: final-refit support count; fold support counts vary'
            else:
                count=metadata[method]['encoded_input_count'] if str(requested)=='all' else int(float(requested))
                semantics='Requested k; all is displayed at final training encoded width, fold widths may vary'
            scores=[]
            for i,(_,validation) in enumerate(splits):
                y=data.y.iloc[train[validation]].to_numpy()
                rmse=-float(candidate[f'split{i}_test_score'])
                sst=float(((y-y.mean())**2).sum())
                scores.append(1-len(y)*rmse**2/sst if sst>0 else np.nan)
            rows.append({'method':method,'display_k':count,'requested_k':str(requested),
                         'count_semantics':semantics,'cv_rmse':-float(candidate.mean_test_score),
                         'cv_rmse_std':float(candidate.std_test_score),
                         'validation_r2':float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan,
                         'validation_r2_std':float(np.nanstd(scores)) if np.isfinite(scores).any() else np.nan,
                         'validation_r2_source':'Reconstructed exactly from recorded per-fold RMSE, n and SST; training-only folds'})
    return pd.DataFrame(rows)


def generate_figures(config,data,run,table,curves,best,figure_dir):
    figures=[]
    def save(fig,name):
        fig.tight_layout();p=figure_dir/(name+'.png');fig.savefig(p,dpi=300,bbox_inches='tight');plt.close(fig)
        figures.append({'name':name,'path':str(p.relative_to(ROOT)).replace('\\','/'),'dpi':300})
    train=np.asarray(run['train_row_indices']);test=np.asarray(run['test_row_indices'])
    unit=config.target_unit;target=f'{config.target} ({unit})';chosen=run['best_model']['method']
    fig,ax=plt.subplots(figsize=(8,5))
    bins=np.arange(-.5,21.5,1) if config.dataset_id=='student_performance' else 45
    ax.hist(data.y.iloc[train],bins=bins,density=True,alpha=.55,label='Training',color=COLORS[1])
    ax.hist(data.y.iloc[test],bins=bins,density=True,alpha=.45,label='Held-out test',color=COLORS[2])
    ax.set(xlabel=target,ylabel='Density',title=f'{config.display_name}: target distributions');ax.legend();save(fig,'target_distribution_train_test')
    numeric=data.X.iloc[train][list(config.numeric_columns)].copy();numeric[config.target]=data.y.iloc[train]
    corr=numeric.corr();fig,ax=plt.subplots(figsize=(12,10))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(len(corr)),corr.columns,rotation=90,fontsize=8);ax.set_yticks(range(len(corr)),corr.index,fontsize=8)
    ax.set_title('Training numeric original inputs + target\nNominal categorical inputs excluded; Pearson correlation')
    fig.colorbar(im,ax=ax,label='Pearson r',shrink=.75);save(fig,'training_original_numeric_correlation_heatmap')
    for metric,error,name,label in [('cv_rmse','cv_rmse_std','feature_count_vs_inner_cv_rmse',f'Mean inner CV RMSE ({unit})'),
                                     ('validation_r2','validation_r2_std','feature_count_vs_inner_validation_r2','Mean inner validation R2')]:
        fig,ax=plt.subplots(figsize=(10,6))
        for method,color in zip(METHODS,COLORS):
            points=curves[curves.method==method].sort_values('display_k')
            ax.errorbar(points.display_k,points[metric],yerr=points[error],marker='o',capsize=3,
                        linestyle='--' if method=='RFECV' else '-',color=color,label=method+(' (best alpha only)' if method=='LASSO' else ''))
        ax.set(xlabel='Encoded feature count (see count semantics in CSV)',ylabel=label,
               title='Recorded training-only CV; sparse FAST grid\nLASSO x is final-refit k; all uses final training width' if run['mode']=='FAST' else 'Recorded training-only CV grid')
        ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);save(fig,name)
    fig,ax=plt.subplots(figsize=(11,5))
    order=list(METHODS);x=np.arange(len(order));meta={r['method']:r for r in run['methods']}
    ax.bar(x-.25,[meta[m]['best_cv_rmse'] for m in order],width=.25,label='Inner CV tuning',color=COLORS[1],yerr=[meta[m]['cv_std'] for m in order])
    ax.bar(x,[meta[m]['nested_cv_rmse'] for m in order],width=.25,label='Outer nested CV',color=COLORS[2],yerr=[meta[m]['nested_cv_std'] for m in order])
    ax.bar(x+.25,[table[(table.method==m)&(table.split=='test')].iloc[0].rmse for m in order],width=.25,label='Held-out test (descriptive)',color='#c4b5fd')
    ax.set_xticks(x,[m.replace(' ','\n') for m in order]);ax.set(ylabel=f'RMSE ({unit})',title=f'Baseline vs feature selection; training-CV winner: {chosen}')
    ax.legend(fontsize=8);save(fig,'baseline_vs_feature_selection_rmse')
    y=best['y_test'];pred=best['pred_test'];res=best['residual'];fit=best['pred_train'];names=best['names'];Z=best['Z_train']
    fig,ax=plt.subplots(figsize=(6,6));ax.scatter(y,pred,s=12,alpha=.35,color=COLORS[2],rasterized=True)
    lo,hi=min(np.min(y),np.min(pred)),max(np.max(y),np.max(pred));ax.plot([lo,hi],[lo,hi],'--',color='#475569')
    ax.set(xlabel='Actual '+target,ylabel='Predicted '+target,title=f'Held-out predictions: {chosen}');save(fig,'heldout_actual_vs_predicted')
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(fit,res,s=10,alpha=.25,color=COLORS[1],rasterized=True)
    bins=best['bins'];ax.plot(bins.fitted_mean,bins.residual_mean,'o-',color=COLORS[3],label='Grouped/bin mean residual')
    ax.axhline(0,color='#475569',ls='--');ax.set(xlabel='Training fitted '+target,ylabel=f'Training residual ({unit})',title=f'Residuals vs fitted: {chosen}');ax.legend();save(fig,'training_residuals_vs_fitted')
    fig,ax=plt.subplots(figsize=(8,5));ax.hist(res,bins=45,density=True,alpha=.6,label='Training',color=COLORS[1])
    ax.hist(y-pred,bins=35,density=True,alpha=.4,label='Held-out test',color=COLORS[2]);ax.set(xlabel=f'Residual ({unit})',ylabel='Density',title='Residual distribution; no normality assumed');ax.legend();save(fig,'residual_histogram_train_test')
    fig,ax=plt.subplots(figsize=(6,6));stats.probplot(res,dist='norm',plot=ax)
    ax.set(xlabel='Theoretical normal quantiles',ylabel=f'Ordered training residuals ({unit})',title=f'Training normal Q-Q: {chosen}');save(fig,'training_residual_normal_qq')
    coeff=best['model']['regressor'].coef_;order_coeff=np.argsort(np.abs(coeff));fig,ax=plt.subplots(figsize=(9,max(4,len(names)*.35)))
    ax.barh(np.arange(len(names)),coeff[order_coeff],color=[COLORS[2] if c>=0 else COLORS[5] for c in coeff[order_coeff]])
    ax.set_yticks(range(len(names)),[names[i].replace('numeric__','').replace('categorical__','') for i in order_coeff],fontsize=9)
    ax.axvline(0,color='#475569');ax.set(xlabel=f'Coefficient ({unit} per encoded input unit)',title=f'Selected coefficients: {chosen}\nAssociation, not causation; numeric inputs standardized');save(fig,'selected_feature_coefficients')
    corr=pd.DataFrame(Z,columns=names).corr();fig,ax=plt.subplots(figsize=(max(6,len(names)*.55),max(5,len(names)*.5)))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(len(names)),names,rotation=90,fontsize=8);ax.set_yticks(range(len(names)),names,fontsize=8)
    ax.set_title('Selected encoded training feature correlation');fig.colorbar(im,ax=ax,shrink=.8);save(fig,'selected_encoded_correlation_heatmap')
    acf_values=best['assumptions'][1]['statistic']['acf'];fig,ax=plt.subplots(figsize=(8,4))
    if acf_values:
        ax.stem(range(len(acf_values)),acf_values,basefmt=' ');band=1.96/np.sqrt(len(res));ax.axhspan(-band,band,color='#94a3b8',alpha=.25,label='Approximate white-noise 95% band')
    ax.set(xlabel='Lag (10-minute observations)' if config.split_strategy=='chronological' else 'Lag in source row order (not time)',ylabel='Residual ACF',title='Training residual autocorrelation'+('' if config.split_strategy=='chronological' else ' (descriptive only)'));ax.legend(fontsize=8);save(fig,'training_residual_autocorrelation')
    intervals=best['intervals'];temporal=config.split_strategy=='chronological'
    display=np.arange(min(288,len(test))) if temporal else np.argsort(pred)
    axis=data.temporal_metadata.date.iloc[test[display]].to_numpy() if temporal else np.arange(len(display))
    fig,ax=plt.subplots(figsize=(11,5))
    ax.fill_between(axis,intervals.individual_pi_lower.iloc[display],intervals.individual_pi_upper.iloc[display],alpha=.2,color=COLORS[1],label='Approximate classical 95% individual PI')
    ax.fill_between(axis,intervals.mean_ci_lower.iloc[display],intervals.mean_ci_upper.iloc[display],alpha=.5,color=COLORS[2],label='Approximate classical 95% mean CI')
    ax.plot(axis,pred[display],color=COLORS[2],lw=1,label='Sklearn point prediction');ax.scatter(axis,y[display],s=9,color='#334155',label='Actual held-out target')
    ax.set(xlabel='First 288 chronological test observations' if temporal else 'Held-out rows sorted by predicted mean (display only)',ylabel=target,title='Pointwise post-selection intervals; iid assumptions may fail')
    ax.legend(fontsize=8,ncol=2)
    if temporal:fig.autofmt_xdate()
    save(fig,'heldout_mean_confidence_and_individual_prediction_intervals')
    return figures


def evaluate_dataset(config,mode='FAST',root=ROOT):
    start=perf_counter();source=root/'results'/config.dataset_id/mode
    run=json.loads((source/'run.json').read_text(encoding='utf-8'));data=load_dataset(config)
    if hashlib.sha256(config.path.read_bytes()).hexdigest()!=run['source_sha256']:raise ValueError('Dataset differs from fitted run')
    train=np.asarray(run['train_row_indices']);test=np.asarray(run['test_row_indices'])
    tables=root/'reports/tables'/config.dataset_id/mode;figures_dir=root/'reports/figures'/config.dataset_id/mode
    tables.mkdir(parents=True,exist_ok=True);figures_dir.mkdir(parents=True,exist_ok=True)
    metric_rows=[];cv_rows=[];all_diagnostics={};coefficient_rows=[];interval_summaries=[];best=None
    temporal=config.split_strategy=='chronological'
    # Source order is essential for temporal residuals; never diagnose a shuffled order.
    residual_order=np.argsort(data.X.index[train])
    for record in run['methods']:
        method=record['method'];slug=method.lower().replace(' ','_')
        path=source/'models'/(slug+'.joblib');model=joblib.load(path)
        Z_train=selected_matrix(model,data.X.iloc[train]);Z_test=selected_matrix(model,data.X.iloc[test])
        pred_train=model.predict(data.X.iloc[train]);pred_test=model.predict(data.X.iloc[test])
        # Confirm the actual saved Phase 2 predictions, not newly invented scores.
        for part,prediction in [('train',pred_train),('test',pred_test)]:
            saved=pd.read_csv(source/(part+'_predictions.csv'))
            np.testing.assert_allclose(saved[method],prediction,rtol=1e-10,atol=1e-7)
        details=feature_details(model,config);names=[d['encoded_name'] for d in details]
        assumptions,vifs,bins,info=assumption_diagnostics(Z_train[residual_order],data.y.iloc[train].to_numpy()[residual_order],pred_train[residual_order],names,temporal=temporal)
        for part,indices,Z,prediction in [('train',train,Z_train,pred_train),('test',test,Z_test,pred_test)]:
            metrics=rank_metrics(data.y.iloc[indices],prediction,info['rank'],design_information(Z)['rank'])
            metric_rows.append({'dataset_id':config.dataset_id,'mode':mode,'method':method,'split':part,
                                'selected_encoded_count':len(details),'original_input_count':config.original_feature_count,**metrics})
        cv_rows.append({'method':method,'inner_cv_rmse':record['best_cv_rmse'],'inner_cv_std':record['cv_std'],
                        'outer_nested_cv_rmse':record['nested_cv_rmse'],'outer_nested_cv_std':record['nested_cv_std'],
                        'selection_basis':'training-only inner CV; test never selects model'})
        vifs.to_csv(tables/(slug+'_vif.csv'),index=False)
        pd.DataFrame(Z_train,columns=names).corr().to_csv(tables/(slug+'_encoded_correlation.csv'))
        intervals,interval_metadata=prediction_intervals(Z_train,data.y.iloc[train].to_numpy(),Z_test,pred_test,temporal=temporal)
        intervals.insert(0,'source_row_index',data.X.index[test]);intervals.insert(1,'actual',data.y.iloc[test].to_numpy())
        if temporal:intervals.insert(1,'date',data.temporal_metadata.date.iloc[test].to_numpy())
        intervals.to_csv(tables/(slug+'_test_intervals.csv'),index=False)
        finite=intervals.individual_pi_lower.notna() & intervals.individual_pi_upper.notna()
        covered=(intervals.actual>=intervals.individual_pi_lower)&(intervals.actual<=intervals.individual_pi_upper)
        interval_summaries.append({'method':method,'estimable_rows':int(intervals.estimable.sum()),
                                   'rows_with_pi':int(finite.sum()),
                                   'empirical_heldout_individual_pi_coverage':float(covered[finite].mean()) if finite.any() else None,
                                   'median_individual_pi_width':float((intervals.individual_pi_upper-intervals.individual_pi_lower)[finite].median()) if finite.any() else None,
                                   'note':'Descriptive coverage on this test set; no interval tuning or future coverage guarantee. Mean CI coverage is not assessed using individual outcomes.'})
        coefficient_rows.append({'method':method,'feature':'Intercept','coefficient':float(model['regressor'].intercept_),'identifiable':bool(info['coefficient_estimable'][0])})
        coefficient_rows.extend({'method':method,'feature':d['encoded_name'],'original_feature':d['original_name'],'coefficient':d['coefficient'],
                                 'identifiable':bool(info['coefficient_estimable'][i+1])} for i,d in enumerate(details))
        rank_info={k:info[k] for k in ('rank','feature_rank','columns_with_intercept','df_resid','singular','condition_number','effective_centered_condition','rank_tolerance')}
        all_diagnostics[method]={'rank':rank_info,'assumptions':assumptions,'intervals':interval_metadata,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if method==run['best_model']['method']:
            best={'model':model,'Z_train':Z_train,'names':names,'pred_train':pred_train,'pred_test':pred_test,
                  'y_test':data.y.iloc[test].to_numpy(),'residual':data.y.iloc[train].to_numpy()-pred_train,
                  'bins':bins,'assumptions':assumptions,'intervals':intervals}
        print(f"{config.dataset_id}: diagnosed {method}, rank={info['rank']}/{info['columns_with_intercept']}",flush=True)
    metrics=pd.DataFrame(metric_rows);metrics.to_csv(tables/'evaluation_metrics.csv',index=False)
    pd.DataFrame(cv_rows).to_csv(tables/'cross_validation_metrics.csv',index=False)
    pd.DataFrame(coefficient_rows).to_csv(tables/'coefficient_identifiability.csv',index=False)
    curves=cv_curve_table(source,config,data,run);curves.to_csv(tables/'training_cv_feature_count_curves.csv',index=False)
    numeric=data.X.iloc[train][list(config.numeric_columns)].copy();numeric[config.target]=data.y.iloc[train]
    numeric.corr().to_csv(tables/'training_original_numeric_correlation.csv')
    figures=generate_figures(config,data,run,metrics,curves,best,figures_dir)
    metadata={'phase':3,'dataset_id':config.dataset_id,'mode':mode,'target':config.target,'target_unit':config.target_unit,
              'source_run':str(source.relative_to(root)).replace('\\','/'),'configuration':run['configuration'],
              'source_run_sha256':hashlib.sha256((source/'run.json').read_bytes()).hexdigest(),
              'selected_method':run['best_model']['method'],'selection_unchanged':True,
              'adjusted_r2_convention':'Training effective affine design rank (including intercept) defines model complexity. Held-out adjusted R2 uses n_test-rank_train as a descriptive penalty, not refitted-test inference; null for n<=rank or constant target.',
              'supersedes':'Only Phase 2 nominal-count adjusted R2 is superseded. Predictions and other metrics are unchanged.',
              'interval_note':'Classical mean CI and individual PI are pointwise working-model intervals. HC3/HAC mean intervals are asymptotic approximations. None correct selection uncertainty or ensure temporal/heteroscedastic PI coverage.',
              'models':all_diagnostics,'figures':figures,'runtime_seconds':perf_counter()-start}
    write_json(tables/'diagnostics.json',metadata)
    pd.DataFrame(interval_summaries).to_csv(tables/'interval_summary.csv',index=False)
    summary_rows=[]
    for method,diagnostic in all_diagnostics.items():
        for entry in diagnostic['assumptions']:
            summary_rows.append({'method':method,'assumption':entry['assumption'],'diagnostic_method':entry['method'],
                                 'statistic_json':json.dumps(sanitize(entry['statistic'])),
                                 'possible_violation':entry['possible_violation'],'interpretation':entry['interpretation'],'limitations':entry['limitations']})
    pd.DataFrame(summary_rows).to_csv(tables/'assumption_summary.csv',index=False)
    lines=[f'# {config.display_name}: Phase 3 ({mode})','',f"Training-selected method: {metadata['selected_method']}. No new model search or frontend changes.",'',
           '## Held-out results (all saved final pipelines)','', '| Method | Encoded count | Effective training rank | RMSE | R2 | Adjusted R2 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for _,r in metrics[metrics.split=='test'].iterrows():
        lines.append(f'| {r.method} | {r.selected_encoded_count} | {r.fitted_design_rank} | {r.rmse:.4f} | {r.r2:.4f} | {r.adjusted_r2:.4f} |')
    lines += ['',metadata['adjusted_r2_convention'],'','## Selected-model assumption evidence','']
    for entry in best['assumptions']:
        lines += [f"### {entry['assumption']}: {entry['method']}",'',entry['interpretation'],'',entry['limitations'],'']
    lines += ['## Interval interpretation','',metadata['interval_note'],
              'Mean CI concerns E[Y|X]; individual PI adds residual variability for one new Y. Nominal coverage is not asserted. Outside-span rows have no estimable interval. Bounds are not clipped; grades outside 0-20 or negative Wh bounds expose the linear working model limitation.',
              '', 'G1/G2 are historical first/second-period grades; this student early-30 scenario excludes them. Including them changes prediction availability and practical use.' if not temporal else 'Energy residuals are chronological. HAC uses a declared 24-lag bandwidth (4 hours at 10-minute sampling), not a guarantee of forecast uncertainty. Serial dependence weakens conventional OLS and BP/normality-test assumptions.',
              '', 'Feature-count validation R2 is reconstructed from saved fold RMSE and exact fold target SST; no test data, model refitting or fabricated validation scores. LASSO displays only its best-alpha point at final-refit support size, not a fabricated k path.',
              '', 'All tests are exploratory after feature/method selection; they are not causal inference or simultaneous guarantees. FAST has only two outer folds; FULL was not rerun.',
              '', 'See PHASE3.md for methodology, statistical references and output definitions.']
    (root/'reports'/(config.dataset_id+'_'+mode+'_diagnostics.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['all',*PHASE2_DATASETS],default='all')
    parser.add_argument('--mode',choices=['FAST','FULL'],default='FAST',type=str.upper)
    args=parser.parse_args();outputs=[]
    with threadpool_limits(limits=1):
        for key,config in PHASE2_DATASETS.items():
            if args.dataset in ('all',key):outputs.append(evaluate_dataset(config,args.mode))
    write_json(ROOT/'reports'/('phase3_'+args.mode+'_manifest.json'),{'phase':3,'mode':args.mode,'datasets':outputs})
    print('Generated Phase 3 reports:',[(r['dataset_id'],len(r['figures'])) for r in outputs],flush=True)

if __name__=='__main__':main()
