"""Phase 2 contracts and leakage/reproducibility checks on authentic observations."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from threadpoolctl import threadpool_limits
from datasets import (STUDENT, ENERGY, load_student_performance, load_appliances_energy,
                      make_preprocessor, holdout_indices, make_cv, DatasetSchemaError, load_dataset)
from feature_selection import CachedRankingSelector, FixedCountSelector
from regression_pipeline import (MODES, METHODS, ExperimentMode, make_pipeline, make_search,
                                 feature_details, feature_grid, fit_training, regression_metrics)
from phase2_experiment import evaluate_holdout

class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.student = load_student_performance()
        cls.energy = load_appliances_energy()
        cls.limits = threadpool_limits(limits=1)
        # Small authentic chronological slice keeps unit-test CPU time low.
        # Full-size real FAST experiments are separately validated from artifacts.
        cls.cache = tempfile.TemporaryDirectory()
        cls.mini_mode = replace(MODES['FAST'], alpha_fractions=(0.1,))
        cls.X = cls.energy.X.iloc[:180]
        cls.y = cls.energy.y.iloc[:180]
        cls.training = fit_training(ENERGY, cls.X, cls.y, mode=cls.mini_mode,
                                    cache_dir=cls.cache.name, jobs=1)

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()
        cls.limits.restore_original_limits()

    def test_energy_schema_predictor_count_and_date(self):
        data = self.energy
        self.assertEqual(data.X.shape, (19735,25))
        self.assertEqual(len(data.raw_columns),29)
        self.assertIn('lights', data.X)
        self.assertTrue({'Appliances','date','rv1','rv2'}.isdisjoint(data.X))
        self.assertTrue(pd.api.types.is_numeric_dtype(data.y))
        original = pd.read_csv(ENERGY.path)
        pd.testing.assert_series_equal(data.temporal_metadata.date, pd.to_datetime(original.date))
        self.assertTrue(data.X.index.equals(data.temporal_metadata.index))

    def test_student_split_uses_42(self):
        actual = holdout_indices(self.student, random_state=MODES['FAST'].seed)
        expected = train_test_split(np.arange(395), test_size=.2, random_state=42)
        for a,b in zip(actual,expected): np.testing.assert_array_equal(a,b)
        self.assertEqual(tuple(map(len,actual)),(316,79))
        self.assertIsInstance(make_cv(STUDENT,2,random_state=42),KFold)

    def test_chronological_holdout_and_both_cv_levels(self):
        train, test = holdout_indices(self.energy,random_state=42)
        np.testing.assert_array_equal(train,np.arange(15788))
        np.testing.assert_array_equal(test,np.arange(15788,19735))
        for mode in MODES.values():
            outer = make_cv(ENERGY,mode.outer_folds,gap=2)
            self.assertIsInstance(outer,TimeSeriesSplit)
            for tr,va in outer.split(self.energy.X.iloc[train]):
                self.assertLess(tr[-1]+2,va[0])
                for itr,iva in make_cv(ENERGY,mode.inner_folds,gap=2).split(tr):
                    self.assertLess(tr[itr[-1]]+2,tr[iva[0]])
        a,b = holdout_indices(self.energy,gap=3)
        self.assertEqual(b[0]-a[-1],4)

    def test_energy_bad_dates_and_excluded_numeric_rejected(self):
        for column,value in [('date','invalid'),('date',None),('rv1','invalid')]:
            frame = pd.read_csv(ENERGY.path)
            frame[column] = frame[column].astype(object)
            frame.loc[0,column] = value
            with tempfile.TemporaryDirectory() as folder:
                p = Path(folder)/'invalid.csv';frame.to_csv(p,index=False)
                with self.assertRaises(DatasetSchemaError):load_dataset(ENERGY,p)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'wrong_order.csv'
            pd.read_csv(ENERGY.path).iloc[::-1].to_csv(p,index=False)
            with self.assertRaisesRegex(DatasetSchemaError,'chronological'):load_dataset(ENERGY,p)

    def test_all_six_pipelines_and_linear_final_estimators(self):
        for data,training_end,test_end in [(self.student,300,395),(self.energy,400,500)]:
            for method in METHODS:
                pipeline = make_pipeline(data.config,method)
                pipeline.fit(data.X.iloc[:training_end],data.y.iloc[:training_end])
                self.assertIsInstance(pipeline['regressor'],LinearRegression)
                self.assertTrue(np.isfinite(pipeline.predict(data.X.iloc[training_end:test_end])).all())
                features = feature_details(pipeline,data.config)
                self.assertTrue(1<=len(features)<=len(pipeline['preprocessor'].get_feature_names_out()))
                self.assertTrue(set(f['original_name'] for f in features).issubset(data.X.columns))

    def test_preprocessor_and_selector_fitted_in_each_inner_fold(self):
        train,heldout = holdout_indices(self.student,random_state=42)
        X,y = self.student.X.iloc[train],self.student.y.iloc[train]
        pre_calls,rank_calls = [],[]
        original_pre = ColumnTransformer.fit_transform
        original_rank = CachedRankingSelector.fit
        def pre_spy(instance,X,*args,**kwargs):
            pre_calls.append(frozenset(X.index))
            return original_pre(instance,X,*args,**kwargs)
        def rank_spy(instance,X,y):
            rank_calls.append(len(X))
            return original_rank(instance,X,y)
        search = make_search(STUDENT,'Correlation',X,MODES['FAST'],jobs=1)
        with patch.object(ColumnTransformer,'fit_transform',pre_spy),patch.object(CachedRankingSelector,'fit',rank_spy):
            search.fit(X,y)
        folds = list(make_cv(STUDENT,2,random_state=42).split(X))
        grid = feature_grid(STUDENT,X,MODES['FAST'])
        expected = [frozenset(X.iloc[tr].index) for tr,va in folds]*len(grid)+[frozenset(X.index)]
        self.assertCountEqual(pre_calls,expected)
        self.assertEqual(rank_calls,[len(rows) for rows in pre_calls])
        self.assertTrue(all(not(set(heldout)&rows) for rows in pre_calls))

    def test_cache_reuses_ranking_but_keeps_fold_boundaries(self):
        X,y = self.X,self.y
        calls=[]
        original=CachedRankingSelector.fit
        def spy(instance,X,y):
            calls.append(len(X));return original(instance,X,y)
        with tempfile.TemporaryDirectory() as folder:
            search=make_search(ENERGY,'Correlation',X,self.mini_mode,memory=Memory(folder,verbose=0),jobs=1)
            with patch.object(CachedRankingSelector,'fit',spy):search.fit(X,y)
        # Two inner fold training matrices + final refit, not k times per fold.
        self.assertEqual(sorted(calls),[60,120,180])

    def test_cached_rfe_path_matches_fixed_k_elimination(self):
        values=make_preprocessor(ENERGY).fit_transform(self.X)
        rank=CachedRankingSelector('RFE').fit(values,self.y)
        for k in [1,5,10,20,25]:
            old=FixedCountSelector('RFE',k).fit(values,self.y)
            self.assertEqual(set(rank.selected_indices_[:k]),set(old.selected_indices_))

    def test_full_grid_explores_all_training_encoded_counts(self):
        for data in (self.student,self.energy):
            X=data.X.iloc[:200]
            width=make_preprocessor(data.config).fit(X).transform(X).shape[1]
            self.assertEqual(feature_grid(data.config,X,MODES['FULL']),list(range(1,width+1)))
        self.assertGreater(MODES['FULL'].outer_folds,MODES['FAST'].outer_folds)
        self.assertGreater(MODES['FULL'].inner_folds,MODES['FAST'].inner_folds)

    def test_feature_names_coefficients_and_order_match_prediction(self):
        for method,model in self.training['models'].items():
            details=feature_details(model,ENERGY)
            all_names=model['preprocessor'].get_feature_names_out().tolist()
            encoded=model['preprocessor'].transform(self.X.iloc[-10:])
            positions=[all_names.index(d['encoded_name']) for d in details]
            prediction=model['regressor'].intercept_+encoded[:,positions]@model['regressor'].coef_
            np.testing.assert_allclose(prediction,model.predict(self.X.iloc[-10:]),rtol=1e-10,atol=1e-7)

    def test_training_api_and_holdout_evaluation_cannot_refit(self):
        import inspect
        self.assertFalse({'X_test','y_test'} & set(inspect.signature(fit_training).parameters))
        before=self.training['best_method']
        params={m:self.training['methods'][m]['best_params'].copy() for m in METHODS}
        with patch.object(LinearRegression,'fit',side_effect=AssertionError('Holdout attempted a fit')):
            normal=evaluate_holdout(self.training,self.energy.X.iloc[180:200],self.energy.y.iloc[180:200])
            altered=evaluate_holdout(self.training,self.energy.X.iloc[180:200]*1000,self.energy.y.iloc[180:200]*-1000)
        self.assertEqual(before,self.training['best_method'])
        self.assertEqual(params,{m:self.training['methods'][m]['best_params'] for m in METHODS})
        self.assertNotEqual(normal['Baseline']['rmse'],altered['Baseline']['rmse'])

    def test_reproducible_selection_and_predictions(self):
        second=fit_training(ENERGY,self.X,self.y,mode=self.mini_mode,cache_dir=self.cache.name,jobs=1)
        self.assertEqual(second['best_method'],self.training['best_method'])
        for method in METHODS:
            self.assertEqual(second['methods'][method]['best_params'],self.training['methods'][method]['best_params'])
            self.assertEqual(second['methods'][method]['best_cv_rmse'],self.training['methods'][method]['best_cv_rmse'])
            np.testing.assert_allclose(second['models'][method].predict(self.X),self.training['models'][method].predict(self.X))

    def test_outer_method_choice_uses_only_inner_scores(self):
        result=self.training
        for workflow in result['workflow_folds']:
            rows=[r for r in result['outer_folds'] if r['outer_fold']==workflow['outer_fold']]
            best=min(rows,key=lambda r:r['inner_cv_rmse'])
            self.assertEqual(workflow['chosen_method'],best['method'])
        self.assertEqual(result['best_method'],min(METHODS,key=lambda m:result['methods'][m]['best_cv_rmse']))

    def test_invalid_jobs_order_alignment_and_nonfinite_predictions(self):
        with self.assertRaises(ValueError):make_search(ENERGY,'RFE',self.X,MODES['FAST'],jobs=8)
        with self.assertRaisesRegex(ValueError,'source order'):fit_training(ENERGY,self.X.iloc[::-1],self.y.iloc[::-1])
        with self.assertRaisesRegex(ValueError,'identical'):fit_training(ENERGY,self.X,self.y.iloc[::-1])
        with self.assertRaisesRegex(ValueError,'nonfinite'):regression_metrics([1,2],[1,np.inf],1)

if __name__=='__main__':unittest.main()
