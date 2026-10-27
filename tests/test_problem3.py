"""Problem 3 production integration and preservation contracts."""
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import OFFICIAL_DATASETS, CAR, load_dataset
from datasets.preprocessing import manufacturer_name
from regression_pipeline import feature_grid, MODES

ROOT=Path(__file__).resolve().parents[1]
class Problem3Tests(unittest.TestCase):
    def test_order_schema_split_and_training_only_selection(self):
        self.assertEqual(list(OFFICIAL_DATASETS),['student_performance','appliances_energy','car_price'])
        data=load_dataset(CAR)
        self.assertEqual((len(data.y),len(data.raw_columns),len(data.X.columns)),(205,26,24))
        run=json.loads((ROOT/'results/car_price/FAST/run.json').read_text())
        tr,te=train_test_split(np.arange(205),test_size=.2,random_state=42,shuffle=True)
        self.assertEqual(run['train_row_indices'],tr.tolist())
        self.assertEqual(run['test_row_indices'],te.tolist())
        self.assertEqual((run['train_size'],run['test_size']),(164,41))
        self.assertEqual(run['best_model']['method'],min(run['methods'],key=lambda m:m['best_cv_rmse'])['method'])
        self.assertTrue(all('car_ID' not in m['selected_original_names'] and 'price' not in m['selected_original_names'] for m in run['methods']))
    def test_manufacturer_grid_and_export_descriptor(self):
        data=load_dataset(CAR)
        self.assertEqual(data.X.CarName.nunique(),147)
        self.assertEqual(data.X.CarName.map(manufacturer_name).nunique(),22)
        width=len(CAR.numeric_columns)+sum((data.X[c].map(manufacturer_name) if c=='CarName' else data.X[c]).nunique() for c in CAR.categorical_columns)
        self.assertEqual(max(feature_grid(CAR,data.X,MODES['FULL'])),width)
        d=json.loads((ROOT/'dashboard/dist/data.json').read_text())['datasets']['car_price']
        self.assertEqual(d['metadata']['problem'],'Problem 3: Car Price Prediction')
        self.assertEqual(d['metadata']['unit'],'USD')
        self.assertEqual(d['preprocessing']['manufacturerColumn'],'CarName')
        self.assertEqual(d['preprocessing']['manufacturerAliases']['vw'],'volkswagen')
        self.assertEqual(d['metadata']['encodedFeatureCount'],73)
        for m in d['models']:
            fitted=joblib.load(ROOT/'results/dashboard/FAST/car_price/models'/(m['id']+'.joblib'))
            self.assertEqual(fitted['regressor'].__class__.__name__,'LinearRegression')
    def test_existing_two_problem_predictions_preserved_by_export(self):
        bundle=json.loads((ROOT/'dashboard/dist/data.json').read_text())
        for key in ['student_performance','appliances_energy']:
            run=json.loads((ROOT/'results'/key/'FAST/run.json').read_text())
            data=load_dataset(OFFICIAL_DATASETS[key]);d=bundle['datasets'][key]
            fitted=joblib.load(ROOT/'results'/key/'FAST/best_model.joblib')
            model=next(m for m in d['models'] if m['id']==d['defaultModelId'])
            np.testing.assert_allclose(model['testPredictions'],fitted.predict(data.X.iloc[run['test_row_indices']]),rtol=1e-10,atol=1e-7)
            self.assertEqual(d['metadata']['originalFeatureCount'],OFFICIAL_DATASETS[key].original_feature_count)
if __name__=='__main__':
    unittest.main()
