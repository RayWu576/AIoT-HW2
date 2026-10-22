"""Rank, estimability, model fidelity and interval regression tests."""
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from datasets import STUDENT,ENERGY,load_dataset
from regression_diagnostics import (selected_matrix,design_information,project_design,rank_metrics,
                                    vif_table,assumption_diagnostics,prediction_intervals,binned_residuals)

ROOT=Path(__file__).resolve().parents[1]

class Phase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=threadpool_limits(limits=1)
        cls.data=load_dataset(STUDENT)
        cls.run_metadata=json.loads((ROOT/'results/student_performance/FAST/run.json').read_text())
        cls.train=cls.run_metadata['train_row_indices'];cls.test=cls.run_metadata['test_row_indices']
        cls.model=joblib.load(ROOT/'results/student_performance/FAST/best_model.joblib')
        cls.Z=selected_matrix(cls.model,cls.data.X.iloc[cls.train])
        cls.new=selected_matrix(cls.model,cls.data.X.iloc[cls.test])
        cls.y=cls.data.y.iloc[cls.train].to_numpy()
        cls.pred=cls.model.predict(cls.data.X.iloc[cls.test])

    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()

    def test_rank_adjusted_r2_agrees_with_ols(self):
        baseline=joblib.load(ROOT/'results/student_performance/FAST/models/baseline.joblib')
        Z=selected_matrix(baseline,self.data.X.iloc[self.train]);info=design_information(Z)
        self.assertEqual((info['rank'],info['columns_with_intercept']),(40,57))
        fit=sm.OLS(self.y,info['basis']).fit()
        metrics=rank_metrics(self.y,baseline.predict(self.data.X.iloc[self.train]),info['rank'])
        self.assertAlmostEqual(metrics['adjusted_r2'],fit.rsquared_adj,places=10)
        old=next(m for m in self.run_metadata['methods'] if m['method']=='Baseline')['train_metrics']['adjusted_r2']
        self.assertNotAlmostEqual(metrics['adjusted_r2'],old,places=3)

    def test_ci_pi_match_statsmodels_on_original_full_rank_design(self):
        intervals,metadata=prediction_intervals(self.Z,self.y,self.new,self.pred)
        design=np.column_stack([np.ones(len(self.Z)),self.Z]);new=np.column_stack([np.ones(len(self.new)),self.new])
        ols=sm.OLS(self.y,design).fit();expected=ols.get_prediction(new).summary_frame()
        for actual,reference in [('mean_ci_lower','mean_ci_lower'),('mean_ci_upper','mean_ci_upper'),('individual_pi_lower','obs_ci_lower'),('individual_pi_upper','obs_ci_upper')]:
            np.testing.assert_allclose(intervals[actual],expected[reference],rtol=1e-10,atol=1e-10)
        self.assertEqual(metadata['residual_df'],len(self.y)-2)
        self.assertTrue((intervals.individual_pi_lower<=intervals.mean_ci_lower).all())
        self.assertTrue((intervals.individual_pi_upper>=intervals.mean_ci_upper).all())

    def test_hc3_mean_ci_matches_reference_and_is_available(self):
        intervals,metadata=prediction_intervals(self.Z,self.y,self.new,self.pred)
        self.assertEqual(metadata['robust_mean_status'],'asymptotic_approximation')
        design=np.column_stack([np.ones(len(self.Z)),self.Z]);new=np.column_stack([np.ones(len(self.new)),self.new])
        expected=sm.OLS(self.y,design).fit().get_robustcov_results(cov_type='HC3',use_t=False).get_prediction(new).summary_frame()
        np.testing.assert_allclose(intervals.robust_mean_ci_lower,expected.mean_ci_lower)
        np.testing.assert_allclose(intervals.robust_mean_ci_upper,expected.mean_ci_upper)

    def test_hac_ci_finite_and_ordered(self):
        intervals,metadata=prediction_intervals(self.Z,self.y,self.new,self.pred,temporal=True,hac_lags=6)
        self.assertEqual(metadata['robust_mean_covariance'],'HAC')
        self.assertTrue(np.isfinite(intervals.robust_mean_ci_lower).all())
        self.assertTrue((intervals.robust_mean_ci_lower<=intervals.robust_mean_ci_upper).all())

    def test_singular_design_masks_nonestimable_rows(self):
        base=load_dataset(ENERGY).X.iloc[:200][['T1','T2']].to_numpy()
        Z=np.column_stack([base,base[:,0]])
        y=load_dataset(ENERGY).y.iloc[:200].to_numpy()
        model=LinearRegression().fit(Z,y)
        new=Z[:10].copy();new[0,-1]+=1
        frame,metadata=prediction_intervals(Z,y,new,model.predict(new))
        self.assertFalse(frame.estimable.iloc[0]);self.assertTrue(frame.estimable.iloc[1:].all())
        self.assertTrue(np.isnan(frame.mean_ci_lower.iloc[0]))
        self.assertEqual(metadata['training_rank'],3)
        self.assertTrue(np.isfinite(frame.individual_pi_upper.iloc[1:]).all())

    def test_vif_constant_perfect_collinearity_and_intercept_handling(self):
        x=np.arange(20,dtype=float)
        table=vif_table(np.column_stack([x,x,np.ones(20)]),['x','copy','constant'])
        self.assertEqual(table.status.tolist(),['perfect_collinearity','perfect_collinearity','constant_column'])
        self.assertTrue(table.vif.isna().all())
        self.assertAlmostEqual(vif_table(x[:,None],['x']).vif.iloc[0],1)

    def test_nonpositive_df_constant_target_and_exact_fit(self):
        Z=np.eye(5);y=np.arange(5,dtype=float)
        frame,metadata=prediction_intervals(Z,y,Z,y)
        self.assertEqual(metadata['status'],'unavailable_nonpositive_residual_df')
        self.assertTrue(frame.mean_ci_lower.isna().all())
        self.assertIsNone(rank_metrics([1,1],[1,1],1)['r2'])
        self.assertIsNone(rank_metrics([1,2],[1,2],2)['adjusted_r2'])
        x=np.arange(10,dtype=float)[:,None];target=2+3*x[:,0]
        frame,metadata=prediction_intervals(x,target,x,target)
        self.assertEqual(metadata['status'],'unavailable_zero_error_variance')

    def test_zero_variance_residual_diagnostics_are_explicit(self):
        x=np.arange(10,dtype=float)[:,None];y=2+3*x[:,0]
        rows,_,_,_=assumption_diagnostics(x,y,y,['x'],temporal=True)
        self.assertIsNone(rows[1]['statistic']['durbin_watson'])
        self.assertIsNone(rows[2]['statistic']);self.assertIsNone(rows[3]['statistic'])

    def test_low_cardinality_fitted_values_retain_all_levels(self):
        fitted=np.array([0,0,0,0,0,1,2,3],dtype=float)
        table=binned_residuals(fitted,np.arange(8,dtype=float))
        self.assertEqual(len(table),4);self.assertEqual(table['count'].sum(),8)

    def test_invalid_designs_and_predictions_raise(self):
        with self.assertRaises(ValueError):design_information(np.array([[np.inf]]))
        with self.assertRaises(ValueError):project_design(design_information(self.Z),np.full_like(self.new,np.nan))
        with self.assertRaises(ValueError):rank_metrics([1,2],[1,np.inf],1)
        with self.assertRaises(ValueError):rank_metrics([],[],1)

if __name__=='__main__':unittest.main()
