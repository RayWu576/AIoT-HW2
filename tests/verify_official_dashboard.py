"""Phase 4: real pipelines versus static browser inference and UI integration."""
import hashlib
import time
import functools
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from playwright.sync_api import sync_playwright
from threadpoolctl import threadpool_limits
from datasets import OFFICIAL_DATASETS,load_dataset
from regression_diagnostics import selected_matrix,prediction_intervals
ROOT=Path(__file__).resolve().parents[1]
DIST=ROOT/'dashboard/dist'
bundle=json.loads((DIST/'data.json').read_text())
started=time.perf_counter()
counts={'models':0,'predictions':0,'interval_rows':0,'missing_unknown_cases':0,'ui_checks':0}
max_prediction_error=0.;max_interval_error=0.
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(ROOT)))
threading.Thread(target=server.serve_forever,daemon=True).start()
with threadpool_limits(limits=1),sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(f'http://127.0.0.1:{server.server_port}/dashboard/dist/index.html')
    assert page.locator('#dataset-select option').count()==2
    for dataset_id,config in OFFICIAL_DATASETS.items():
        d=bundle['datasets'][dataset_id];data=load_dataset(config)
        run=json.loads((ROOT/'results'/dataset_id/'FAST/run.json').read_text())
        tr,te=run['train_row_indices'],run['test_row_indices'];X=data.X.iloc[te];y=data.y.iloc[tr]
        assert d['metadata']['originalFeatureCount']==config.original_feature_count
        assert 14<=config.original_feature_count<=39
        assert {m['method'] for m in d['models']}=={'Baseline','Correlation','Mutual Information','RFE','RFECV','LASSO'}
        raw=json.loads(X.to_json(orient='records'))
        all_js=page.evaluate("""({id,rows})=>{const d=DASHBOARD_DATA.datasets[id];return d.models.map(m=>rows.map(r=>DashboardPrediction.predict(d,m,r)));}""",{'id':dataset_id,'rows':raw})
        for m,js in zip(d['models'],all_js):
            fitted=joblib.load(ROOT/'results/dashboard/FAST'/dataset_id/'models'/(m['id']+'.joblib'))
            expected=fitted.predict(X);np.testing.assert_allclose(js,expected,rtol=1e-10,atol=1e-7)
            np.testing.assert_allclose(m['testPredictions'],expected,rtol=1e-12,atol=1e-10)
            max_prediction_error=max(max_prediction_error,float(np.max(np.abs(np.array(js)-expected))))
            assert np.isclose(np.mean((data.y.iloc[te]-expected)**2)**.5,m['test']['rmse'])
            assert len(m['features'])==m['k'];counts['models']+=1;counts['predictions']+=len(X)
            cases=X.iloc[:3].copy();cases.iloc[0,cases.columns.get_loc(config.numeric_columns[0])]=np.nan
            if config.categorical_columns:
                cases.iloc[1,cases.columns.get_loc(config.categorical_columns[0])]=np.nan
                cases.iloc[2,cases.columns.get_loc(config.categorical_columns[0])]='unseen-category'
            js_cases=page.evaluate("""({id,model,rows})=>{const d=DASHBOARD_DATA.datasets[id],m=d.models.find(m=>m.id===model);return rows.map(r=>DashboardPrediction.predict(d,m,r));}""",{'id':dataset_id,'model':m['id'],'rows':json.loads(cases.to_json(orient='records'))})
            np.testing.assert_allclose(js_cases,fitted.predict(cases),rtol=1e-10,atol=1e-7);counts['missing_unknown_cases']+=3
            if m['isOfficial']:
                js_intervals=page.evaluate("""({id,model,rows})=>{const d=DASHBOARD_DATA.datasets[id],m=d.models.find(m=>m.id===model);return rows.map(r=>DashboardPrediction.intervals(d,m,r));}""",{'id':dataset_id,'model':m['id'],'rows':raw})
                frame,_=prediction_intervals(selected_matrix(fitted,data.X.iloc[tr]),y,selected_matrix(fitted,X),expected,temporal=dataset_id=='appliances_energy')
                assert [r['estimable'] for r in js_intervals]==frame.estimable.tolist()
                for jskey,pykey in [('meanLower','mean_ci_lower'),('meanUpper','mean_ci_upper'),('piLower','individual_pi_lower'),('piUpper','individual_pi_upper'),('robustMeanLower','robust_mean_ci_lower'),('robustMeanUpper','robust_mean_ci_upper')]:
                    actual=np.array([r.get(jskey,np.nan) if r.get(jskey) is not None else np.nan for r in js_intervals]);target=frame[pykey].to_numpy();np.testing.assert_allclose(actual,target,rtol=1e-8,atol=1e-6,equal_nan=True)
                    finite=np.isfinite(target)
                    if finite.any():max_interval_error=max(max_interval_error,float(np.max(np.abs(actual[finite]-target[finite]))))
                counts['interval_rows']+=len(X)
        page.select_option('#dataset-select',dataset_id)
        assert page.locator('#scenarios .scenario').count()==0
        assert str(config.original_feature_count)+' original predictors' in page.locator('#dataset-summary').inner_text()
        assert page.locator('#inputs select').count()==len(config.categorical_columns)
        assert page.locator('#ranking tr').count()==6
        for method in ['Baseline','Correlation','Mutual Information','RFE','RFECV','LASSO']:
            page.select_option('#method-select',method)
            expected_models=[m for m in d['models'] if m['method']==method]
            assert page.locator('#counts button').count()==len(expected_models)
            for m in expected_models:
                page.locator('#counts button[data-model="'+m['id']+'"]').click()
                assert page.locator('#coefficients tr').count()==m['k']+1
                assert page.locator('#input-errors').inner_text()==''
                assert page.locator('#profit').inner_text()
                assert page.locator('#diagnostics svg').count()==5
                counts['ui_checks']+=1
        page.locator('#rmse-chart circle').last.click()
        page.locator('#rmse-chart circle').first.press('Enter')
        page.locator('#compare').uncheck();assert page.locator('#rmse-chart .legend button').count()==1
        page.locator('#compare').check();assert page.locator('#rmse-chart .legend button').count()==6
        page.select_option('#metric-split','train');assert 'TRAIN RMSE' in page.locator('#metrics').inner_text()
        page.select_option('#metric-split','test');assert 'TEST RMSE' in page.locator('#metrics').inner_text()
        page.locator('#save-scenario').click();assert page.locator('#scenarios .scenario').count()==1
        numeric=page.locator('#inputs input[type=number]').first;old=numeric.input_value();numeric.fill('');assert page.locator('#save-scenario').is_disabled();assert page.locator('#input-errors').inner_text();numeric.fill(old)
        numeric.fill(str(float(old)+0.5));assert page.locator('#save-scenario').is_disabled() if dataset_id=='student_performance' else True
        field=next(f for f in d['inputs'] if f['kind']=='numeric')
        numeric.fill(str(field['max']+1));assert 'outside observed training range' in page.locator('#extrapolation').inner_text()
        numeric.fill(old)
        assert '95%' in page.locator('#prediction-intervals').inner_text() or 'not estimable' in page.locator('#prediction-intervals').inner_text()
        if config.categorical_columns:
            result=page.evaluate("""({id,raw})=>{const d=DASHBOARD_DATA.datasets[id],m=d.models.find(m=>m.method==='Baseline');return DashboardPrediction.intervals(d,m,raw);} """,{'id':dataset_id,'raw':json.loads(cases.to_json(orient='records'))[2]})
            assert not result['estimable']
        with page.expect_download() as event:page.locator('#download').click()
        dl=event.value;content=Path(dl.path()).read_text(encoding='utf-8-sig');assert dataset_id in dl.suggested_filename;assert '"dataset_id","'+dataset_id+'"' in content;assert '"actual","predicted"' in content;assert len(content.splitlines())>len(X)
        for width in [1440,768,390,320]:
            page.set_viewport_size({'width':width,'height':900})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(dataset_id,width,page.evaluate('document.documentElement.scrollWidth'))
        page.set_viewport_size({'width':1440,'height':1000});page.screenshot(path=str(ROOT/'reports'/('phase4_'+dataset_id+'_desktop.png')),full_page=True)
        page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(ROOT/'reports'/('phase4_'+dataset_id+'_mobile.png')),full_page=True)
        page.set_viewport_size({'width':1440,'height':1000})
    page.locator('#theme').click();assert page.locator('body').evaluate('(e)=>e.classList.contains("dark")');page.reload();assert page.locator('body').evaluate('(e)=>e.classList.contains("dark")');page.locator('#theme').click()
    page.goto((DIST/'index.html').as_uri());assert page.locator('#dataset-select option').count()==2
    page.route('**/data.js*',lambda route:route.abort());page.reload();assert 'Dashboard data unavailable' in page.locator('main').inner_text()
    assert not errors,errors
    browser.close()
server.shutdown()
report={'passed':True,'runtime_seconds':time.perf_counter()-started,'asset_sha256':{n:hashlib.sha256((DIST/n).read_bytes()).hexdigest() for n in ['app.js','prediction.js','styles.css','index.html','data.json','data.js']},'counts':counts,'max_prediction_absolute_error':max_prediction_error,'max_interval_absolute_error':max_interval_error,'javascript_errors':errors,'viewports':[1440,768,390,320],'deployment_verified':False}
(ROOT/'reports/phase4_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
