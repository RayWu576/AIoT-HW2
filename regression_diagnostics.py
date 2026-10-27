"""Rank-aware diagnostics and intervals for already-fitted regression pipelines."""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf


def selected_matrix(model, X):
    values = model['preprocessor'].transform(X)
    for step in ('ranking', 'selector'):
        if step in model.named_steps: values = model[step].transform(values)
    return np.asarray(values, dtype=float)


def design_information(Z):
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or not np.isfinite(Z).all() or len(Z) == 0:
        raise ValueError('Design must be a nonempty finite two-dimensional matrix')
    center = Z.mean(axis=0)
    centered = Z-center
    u,s,vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape)*np.finfo(float).eps*(s[0] if len(s) else 0)
    q = int(np.sum(s > tolerance))
    vectors, scales = vt[:q], s[:q]
    basis = np.column_stack([np.ones(len(Z)), u[:,:q]])
    full = np.column_stack([np.ones(len(Z)), Z])
    _,full_s,full_vt = np.linalg.svd(full, full_matrices=False)
    # Centered affine rank defines effective slope df plus one intercept.
    rank = q+1
    coefficient_estimable = np.sum(full_vt[:rank]**2,axis=0) >= 1-1e-8
    return {'center': center, 'vectors': vectors, 'scales': scales, 'basis': basis,
            'rank': rank, 'feature_rank': q, 'columns_with_intercept': full.shape[1],
            'df_resid': len(Z)-rank, 'singular': rank < full.shape[1],
            'condition_number': float(full_s[0]/full_s[-1]) if rank == full.shape[1] and full_s[-1]>0 else None,
            'effective_centered_condition': float(scales[0]/scales[-1]) if q else None,
            'coefficient_estimable': coefficient_estimable, 'rank_tolerance': float(tolerance)}


def project_design(info, Z):
    Z=np.asarray(Z,dtype=float)
    if Z.ndim!=2 or Z.shape[1]!=len(info["center"]) or not np.isfinite(Z).all():
        raise ValueError("New design must be finite and match the training columns")
    centered = Z-info['center']
    vectors = info['vectors']
    scores = centered @ vectors.T
    projected = scores @ vectors
    estimable = np.linalg.norm(centered-projected,axis=1) <= 1e-8*(1+np.linalg.norm(centered,axis=1))
    normalized = scores/info['scales'] if len(info['scales']) else scores
    return np.column_stack([np.ones(len(centered)), normalized]), estimable


def rank_metrics(y, predicted, fitted_rank, evaluation_design_rank=None):
    y,predicted = np.asarray(y,dtype=float),np.asarray(predicted,dtype=float)
    if len(y)==0 or len(y)!=len(predicted) or not np.isfinite(y).all() or not np.isfinite(predicted).all():
        raise ValueError('Targets and predictions must be aligned and finite')
    n=len(y); residual=y-predicted
    sse=float(residual@residual); sst=float(((y-y.mean())**2).sum())
    r2=1-sse/sst if sst>0 else None
    adjusted=1-(sse/(n-fitted_rank))/(sst/(n-1)) if n>fitted_rank and n>1 and sst>0 else None
    return {'n':n,'mae':float(np.abs(residual).mean()),'mse':sse/n,'rmse':float(np.sqrt(sse/n)),
            'r2':r2,'adjusted_r2':adjusted,'fitted_design_rank':int(fitted_rank),
            'evaluation_design_rank':evaluation_design_rank,
            'adjustment_denominator':n-int(fitted_rank)}


def vif_table(Z, names):
    values=np.asarray(Z,dtype=float); rows=[]
    for j,name in enumerate(names):
        target=values[:,j]; total=float(((target-target.mean())**2).sum())
        if total<=1e-20:
            rows.append({'feature':name,'vif':None,'status':'constant_column'});continue
        others=np.column_stack([np.ones(len(values)),np.delete(values,j,axis=1)])
        residual=target-others@np.linalg.lstsq(others,target,rcond=None)[0]
        error=float(residual@residual)
        perfect=error<=1e-12*total
        rows.append({'feature':name,'vif':None if perfect else max(1.0,total/error),
                     'status':'perfect_collinearity' if perfect else 'estimable'})
    return pd.DataFrame(rows)


def binned_residuals(fitted,residual):
    frame=pd.DataFrame({'fitted':fitted,'residual':residual})
    if frame.fitted.nunique()<=10:
        return frame.groupby('fitted',sort=True).agg(fitted_mean=('fitted','mean'),residual_mean=('residual','mean'),count=('residual','size')).reset_index(drop=True)
    bins=pd.qcut(frame.fitted,q=min(10,len(frame)),duplicates='drop')
    if bins.isna().all(): return pd.DataFrame(columns=['fitted_mean','residual_mean','count'])
    return frame.groupby(bins,observed=True).agg(fitted_mean=('fitted','mean'),residual_mean=('residual','mean'),count=('residual','size')).reset_index(drop=True)


def assumption_diagnostics(Z,y,predicted,names,*,temporal=False,hac_lags=24):
    info=design_information(Z); residual=np.asarray(y)-np.asarray(predicted)
    variable=float(np.var(residual))>1e-20
    ols=sm.OLS(y,info['basis']).fit() if info['df_resid']>0 else None
    bins=binned_residuals(predicted,residual)
    spread=float(np.ptp(bins.residual_mean)/np.std(residual)) if variable and len(bins)>1 else None
    visual={'bin_count':len(bins),'minimum_bin_mean':float(bins.residual_mean.min()) if len(bins) else None,
            'maximum_bin_mean':float(bins.residual_mean.max()) if len(bins) else None,'bin_mean_span_residual_sd':spread}
    rows=[]
    rows.append({'assumption':'linearity','method':'Binned residual means vs fitted (descriptive)',
                 'statistic':visual,'possible_violation':True if spread is not None and spread>.5 else None,
                 'interpretation':f'Bin-mean residual span is {spread:.3f} residual SD; inspect the plotted trend.' if spread is not None else 'Insufficient distinct fitted values for a trend.',
                 'limitations':'A span >0.5 SD is a declared visual screening heuristic, not a significance test. A small span does not establish linearity.'})
    dw=float(durbin_watson(residual)) if variable else None
    lags=[lag for lag in (1,6,12,24) if lag<len(residual)//4]
    lb=acorr_ljungbox(residual,lags=lags,return_df=True,model_df=0) if temporal and variable and lags else None
    acf_values=acf(residual,nlags=min(40,len(residual)-1),fft=True).tolist() if variable else []
    lb_records=[{'lag':int(lag),'statistic':float(r.lb_stat),'pvalue':float(r.lb_pvalue)} for lag,r in lb.iterrows()] if lb is not None else []
    rows.append({'assumption':'independence','method':'Durbin-Watson and Ljung-Box' if temporal else 'Durbin-Watson (source row order; descriptive only)',
                 'statistic':{'durbin_watson':dw,'ljung_box':lb_records,'acf':acf_values},
                 'possible_violation':bool(any(r['pvalue']<.05 for r in lb_records)) if temporal else None,
                 'interpretation':f'DW={dw:.4f}; lag-1 ACF={acf_values[1]:.4f}.' if dw is not None and len(acf_values)>1 else 'Undefined for constant/degenerate residuals.',
                 'limitations':'Energy residuals retain chronological order; Ljung-Box model_df=0 is exploratory after regression/selection, with multiple correlated lag tests.' if temporal else 'These observations have no justified temporal sampling order. DW cannot establish independence; group dependence remains possible.'})
    bp=None
    if variable and ols is not None and info['rank']>1 and info['df_resid']>1:
        lm,pvalue,fstat,fp=het_breuschpagan(residual,info['basis'],robust=True)
        bp={'lm_statistic':float(lm),'lm_pvalue':float(pvalue),'f_statistic':float(fstat),'f_pvalue':float(fp),'auxiliary_df':info['rank']-1}
    rows.append({'assumption':'homoscedasticity','method':'Koenker-Breusch-Pagan on full-rank equivalent design',
                 'statistic':bp,'possible_violation':bool(bp['lm_pvalue']<.05) if bp else None,
                 'interpretation':f"LM={bp['lm_statistic']:.4f}, p={bp['lm_pvalue']:.3g}; {'evidence against constant variance' if bp['lm_pvalue']<.05 else 'constant variance not rejected'}." if bp else 'Test unavailable: insufficient rank/df or degenerate residuals.',
                 'limitations':'Rank-reduced auxiliary design includes an intercept. Nonrejection is not proof; temporal dependence and selection can invalidate nominal p-values.'})
    normal=None
    if variable and len(residual)>=3:
        test=stats.shapiro(residual) if len(residual)<=5000 else stats.jarque_bera(residual)
        normal={'test':'Shapiro-Wilk' if len(residual)<=5000 else 'Jarque-Bera (large-sample)',
                'statistic':float(test.statistic),'pvalue':float(test.pvalue),'pvalue_underflow':bool(test.pvalue==0),'n':len(residual),
                'skewness':float(stats.skew(residual,bias=False)),'pearson_kurtosis':float(stats.kurtosis(residual,fisher=False,bias=False))}
    normal_p='below floating-point resolution' if normal and normal['pvalue']==0 else f"{normal['pvalue']:.3g}" if normal else 'unavailable'
    rows.append({'assumption':'normality','method':normal['test'] if normal else 'Unavailable',
                 'statistic':normal,'possible_violation':bool(normal['pvalue']<.05) if normal else None,
                 'interpretation':f"p={normal_p}, skew={normal['skewness']:.3f}, kurtosis={normal['pearson_kurtosis']:.3f}; {'normality rejected' if normal['pvalue']<.05 else 'normality not rejected'}." if normal else 'Too few observations or zero residual variance.',
                 'limitations':'Shapiro p-values are not used beyond 5000 observations. Large-sample tests detect small departures; fitted/selected and serially dependent residuals are not iid raw samples. Inspect histogram and Q-Q.'})
    vifs=vif_table(Z,names)
    perfect=int((vifs.status=='perfect_collinearity').sum()); high=int((vifs.vif.fillna(0)>10).sum())
    rows.append({'assumption':'multicollinearity','method':'Design rank, condition number and VIF (intercept excluded)',
                 'statistic':{'rank':info['rank'],'columns_with_intercept':info['columns_with_intercept'],
                              'condition_number':info['condition_number'],'effective_centered_condition':info['effective_centered_condition'],
                              'perfectly_collinear_columns':perfect,'vif_above_10':high},
                 'possible_violation':bool(info['singular'] or high),
                 'interpretation':f"Design rank {info['rank']}/{info['columns_with_intercept']}; {perfect} perfectly collinear columns; {high} finite VIF values above 10.",
                 'limitations':'VIF >10 is a screening convention, not a universal cutoff. Full one-hot sets create exact aliasing; null VIF plus status denotes infinity/undefined, not zero. Individual aliased coefficients are not unique.'})
    return rows,vifs,bins,info


def prediction_intervals(Z_train,y_train,Z_new,sklearn_prediction,*,temporal=False,hac_lags=24,alpha=.05):
    info=design_information(Z_train); B,estimable=project_design(info,Z_new)
    n_new=len(B); nan=np.full(n_new,np.nan)
    frame=pd.DataFrame({'prediction':np.asarray(sklearn_prediction),'estimable':estimable,
                        'mean_ci_lower':nan.copy(),'mean_ci_upper':nan.copy(),
                        'individual_pi_lower':nan.copy(),'individual_pi_upper':nan.copy(),
                        'robust_mean_ci_lower':nan.copy(),'robust_mean_ci_upper':nan.copy()})
    metadata={'level':1-alpha,'training_rank':info['rank'],'residual_df':info['df_resid'],
              'classical_label':'Approximate post-selection OLS t mean CI / individual PI; iid homoscedastic normal-error working model',
              'robust_mean_covariance':'HAC' if temporal else 'HC3','hac_maxlags':hac_lags if temporal else None,
              'limitations':'Pointwise, conditional on selected features; excludes feature/method-selection uncertainty. Classical PI is not heteroscedasticity/autocorrelation-robust. Robust mean CI does not provide an individual PI or forecast guarantee. No clipping to physical target bounds.'}
    if info['df_resid']<=0:
        metadata['status']='unavailable_nonpositive_residual_df';return frame,metadata
    ols=sm.OLS(y_train,info['basis']).fit()
    matched=ols.predict(B)
    np.testing.assert_allclose(matched,sklearn_prediction,rtol=1e-7,atol=1e-7,
                               err_msg='Statsmodels design is inconsistent with sklearn predictions')
    metadata['maximum_mean_prediction_difference']=float(np.max(np.abs(matched-sklearn_prediction)))
    if float(ols.ssr)<=1e-20*max(1,float(np.sum(np.asarray(y_train)**2))):
        metadata['status']='unavailable_zero_error_variance';return frame,metadata
    sf=ols.get_prediction(B).summary_frame(alpha=alpha)
    for destination,source in [('mean_ci_lower','mean_ci_lower'),('mean_ci_upper','mean_ci_upper'),
                               ('individual_pi_lower','obs_ci_lower'),('individual_pi_upper','obs_ci_upper')]:
        frame.loc[estimable,destination]=np.asarray(sf[source])[estimable]
    robust=None
    if temporal:
        robust=ols.get_robustcov_results(cov_type='HAC',maxlags=min(hac_lags,len(y_train)-1),use_correction=True,use_t=False)
    elif np.max(1/len(y_train)+np.sum(info['basis'][:,1:]**2,axis=1))<1-1e-10:
        robust=ols.get_robustcov_results(cov_type='HC3',use_t=False)
    if robust is not None and np.isfinite(robust.cov_params()).all():
        variance=np.einsum('ij,jk,ik->i',B,robust.cov_params(),B)
        half=stats.norm.ppf(1-alpha/2)*np.sqrt(np.maximum(0,variance))
        frame.loc[estimable,'robust_mean_ci_lower']=(matched-half)[estimable]
        frame.loc[estimable,'robust_mean_ci_upper']=(matched+half)[estimable]
        metadata['robust_mean_status']='asymptotic_approximation'
    else:metadata['robust_mean_status']='unavailable_high_leverage_or_nonfinite_covariance'
    metadata['status']='estimated' if estimable.all() else 'partly_nonestimable_outside_training_affine_span'
    metadata['nonestimable_rows']=int((~estimable).sum())
    return frame,metadata
