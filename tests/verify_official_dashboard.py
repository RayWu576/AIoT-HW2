"""Integration QA for real fitted models, browser predictions and existing UI."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright
from datasets import OFFICIAL_DATASETS, load_dataset
from regression_experiment import METHODS

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dashboard/dist'
bundle = json.loads((DIST/'data.json').read_text(encoding='utf-8'))
assert set(bundle['datasets']) == set(OFFICIAL_DATASETS)
counts = {'curve_models': 0, 'row_predictions': 0, 'official_pipelines': 0}

with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width':1440, 'height':1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto((DIST/'index.html').as_uri())
    assert page.locator('#dataset-select option').count() == 2
    for dataset_id, config in OFFICIAL_DATASETS.items():
        payload = bundle['datasets'][dataset_id]
        data = load_dataset(config)
        metadata = json.loads((ROOT/'results'/dataset_id/'metadata.json').read_text(encoding='utf-8'))
        train, test = metadata['train_indices'], metadata['test_indices']
        assert not set(train) & set(test)
        assert len(train) + len(test) == len(data.y)
        assert payload['metadata']['originalFeatureCount'] == config.original_feature_count
        assert {m['method'] for m in payload['models']} == set(METHODS)
        comparison = pd.read_csv(ROOT/'results'/dataset_id/'model_comparison.csv')
        assert list(comparison['CV RMSE']) == sorted(comparison['CV RMSE'])
        recommendation = comparison[comparison.Model != 'Baseline'].iloc[0]
        assert (payload['defaultMethod'], payload['defaultK']) == (recommendation.Model, recommendation.Features)
        raw = data.X.iloc[test].to_dict(orient='records')
        js_predictions = page.evaluate("""({id,rows})=>{
            const d=window.DASHBOARD_DATA.datasets[id];
            return d.models.map(m=>rows.map(r=>DashboardPrediction.predict(d,m,r)));
        }""", {'id':dataset_id, 'rows':raw})
        for model, predicted in zip(payload['models'], js_predictions):
            actual = np.array([r['actual'] for r in model['residuals']])
            expected = np.array([r['predicted'] for r in model['residuals']])
            np.testing.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-7)
            assert np.isclose(np.sqrt(np.mean((actual-expected)**2)), model['rmse'])
            assert len(model['features']) == model['k']
            counts['curve_models'] += 1
            counts['row_predictions'] += len(predicted)
        # Compare exported equations directly with each official fitted sklearn Pipeline,
        # including missing numerics, missing categories and unseen manufacturers.
        fitted_models = joblib.load(ROOT/'results'/dataset_id/'models.pkl')
        cases = data.X.iloc[test[:4]].copy()
        cases.loc[cases.index[0], config.numeric_columns[0]] = np.nan
        cases.loc[cases.index[1], config.categorical_columns[0]] = np.nan
        cases.loc[cases.index[2], config.categorical_columns[0]] = 'unseen brand model'
        if dataset_id == 'car_price': cases.loc[cases.index[3], 'CarName'] = 'vw golf'
        # JSON uses null for missing values.
        records = json.loads(cases.to_json(orient='records'))
        for method in METHODS:
            fitted = fitted_models[method]
            k = int(fitted['selector'].get_support().sum())
            js = page.evaluate("""({id,method,k,rows})=>{
                const d=window.DASHBOARD_DATA.datasets[id],m=d.models.find(m=>m.method===method&&m.k===k);
                return rows.map(r=>DashboardPrediction.predict(d,m,r));
            }""", {'id':dataset_id,'method':method,'k':k,'rows':records})
            np.testing.assert_allclose(js, fitted.predict(cases), rtol=1e-10, atol=1e-7)
            counts['official_pipelines'] += 1
        page.select_option('#dataset-select',dataset_id)
        assert str(config.original_feature_count)+' original inputs' in page.locator('#dataset-summary').inner_text()
        assert page.locator('#inputs select').count() == len(config.categorical_columns)
        for method in METHODS:
            page.locator(f'#methods button[data-method="{method}"]').click()
            page.locator('#counts button[data-k="3"]').click()
            model = next(m for m in payload['models'] if m['method']==method and m['k']==3)
            prefix = '$' if dataset_id == 'car_price' else ''
            assert page.locator('#metrics .metric strong').first.inner_text() == f"{prefix}{model['rmse']:,.2f}"
            assert page.locator('#coefficients tr').count() == 4
            raw_defaults = {f['name']:f['default'] for f in payload['inputs']}
            expected = page.evaluate("""({id,method,raw})=>{
                const d=window.DASHBOARD_DATA.datasets[id],m=d.models.find(m=>m.method===method&&m.k===3);
                return DashboardPrediction.predict(d,m,raw);
            }""", {'id':dataset_id,'method':method,'raw':raw_defaults})
            assert page.locator('#profit').inner_text() == f'{prefix}{expected:,.2f}'
        if dataset_id == 'car_price':
            brand_index = next(i for i,f in enumerate(payload['inputs']) if f['name']=='CarName')
            symbol_index = next(i for i,f in enumerate(payload['inputs']) if f['name']=='symboling')
            assert len(payload['inputs'][brand_index]['options']) <= 22
            page.select_option(f'#category-{brand_index}', 'volkswagen')
            page.locator(f'#number-{symbol_index}').fill('-2')
            assert 'NaN' not in page.locator('#profit').inner_text()
        page.locator('#save-scenario').click()
        assert page.locator('#scenarios .scenario').count() == 1
        with page.expect_download() as download: page.locator('#download').click()
        assert download.value.suggested_filename.startswith(dataset_id+'-')
        page.locator('#compare').uncheck()
        assert page.locator('#rmse-chart circle').count() == len([m for m in payload['models'] if m['method']=='LASSO'])
        page.select_option('#sort','id')
        first = page.locator('#residuals td:first-child').first.inner_text()
        page.locator('#next-page').click()
        assert first != page.locator('#residuals td:first-child').first.inner_text()
        page.locator('#theme').click()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'Mobile overflow'
        qa = ROOT/'dashboard/qa';qa.mkdir(exist_ok=True)
        page.screenshot(path=str(qa/(dataset_id+'-mobile.png')), full_page=True)
        page.set_viewport_size({'width':1440,'height':1000})
        page.screenshot(path=str(qa/(dataset_id+'-desktop.png')), full_page=True)
    page.select_option('#dataset-select', 'student_performance')
    assert page.locator('#scenarios .scenario').count() == 0
    assert '$' not in page.locator('#profit').inner_text()
    assert not errors, errors
    browser.close()
(ROOT/'results/official_validation.json').write_text(json.dumps({'status':'PASS',**counts},indent=2),encoding='utf-8')
print('PASS:', counts, 'two-dataset switching, inputs, units, CSV, scenarios, pagination, mobile width; no JS errors')
