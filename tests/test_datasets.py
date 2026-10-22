"""Official dataset contract tests using authentic CSVs and explicitly mutated test cases."""
import tempfile
from dataclasses import replace
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from datasets import (CAR, DatasetSchemaError, load_dataset, load_student_performance,
                      load_car_price, make_preprocessor, feature_counts,
                      student_config, holdout_indices, make_cv)

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.student = load_student_performance(include_prior_grades=False)
        cls.full = load_student_performance(include_prior_grades=True)
        cls.car = load_car_price()

    def mutated(self, config, mutate):
        frame = pd.read_csv(config.path, sep=config.separator)
        mutate(frame)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / config.filename
            frame.to_csv(path, sep=config.separator, index=False)
            return load_dataset(config, path)

    def test_student_30_explicit_early_profile(self):
        data = self.student
        self.assertEqual(data.X.shape, (395, 30))
        self.assertEqual(len(data.raw_columns), 33)
        self.assertEqual(len(data.config.numeric_columns), 13)
        self.assertEqual(len(data.config.categorical_columns), 17)
        self.assertTrue({'G1', 'G2'}.isdisjoint(data.X.columns))

    def test_authentic_student_32_preserves_prior_grades(self):
        self.assertEqual(self.full.X.shape, (395, 32))
        self.assertTrue({'G1', 'G2'}.issubset(self.full.X.columns))
        self.assertEqual(load_student_performance().original_feature_count, 30)

    def test_car_24_candidate_inputs(self):
        self.assertEqual(self.car.X.shape, (205, 24))
        self.assertEqual(len(self.car.raw_columns), 26)
        self.assertIn('CarName', self.car.X)
        self.assertTrue({'car_ID', 'price'}.isdisjoint(self.car.X))
        self.assertEqual(CAR.target_unit, 'USD')
        self.assertEqual((len(CAR.numeric_columns), len(CAR.categorical_columns)), (14, 10))

    def test_targets_and_feature_contract(self):
        for data in (self.student, self.full, self.car):
            self.assertNotIn(data.config.target, data.X)
            self.assertTrue(pd.api.types.is_numeric_dtype(data.y))
            self.assertFalse(data.y.isna().any())
            self.assertTrue(14 <= data.original_feature_count <= 39)
            self.assertEqual(tuple(data.X.columns), data.config.feature_names)
            self.assertEqual(set(data.X), set(data.config.numeric_columns) | set(data.config.categorical_columns))

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, 'acquisition instructions'):
                load_car_price(Path(folder) / 'absent.csv')

    def test_missing_extra_and_duplicate_headers(self):
        for config in (self.student.config, CAR):
            with self.assertRaisesRegex(DatasetSchemaError, 'missing='):
                self.mutated(config, lambda f: f.drop(columns=config.feature_names[0], inplace=True))
            with self.assertRaisesRegex(DatasetSchemaError, 'unexpected='):
                self.mutated(config, lambda f: f.insert(0, 'unwanted', 1))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'duplicate.csv'
                header = list(config.raw_columns)
                header[-1] = header[0]
                path.write_text(config.separator.join(header), encoding='utf-8')
                with self.assertRaisesRegex(DatasetSchemaError, 'duplicate'):
                    load_dataset(config, path)

    def test_bad_numeric_including_excluded_columns(self):
        for config, column in ((self.student.config, 'age'), (self.student.config, 'G1'), (CAR, 'car_ID'), (CAR, 'price')):
            def mutate(frame):
                frame[column] = frame[column].astype(object)
                frame.loc[0, column] = 'invalid'
            with self.assertRaisesRegex(DatasetSchemaError, column + ' must be numeric'):
                self.mutated(config, mutate)

    def test_missing_target_and_infinity_rejected(self):
        for config in (self.student.config, CAR):
            with self.assertRaisesRegex(DatasetSchemaError, 'target .* missing'):
                self.mutated(config, lambda f: f.__setitem__(config.target, np.nan))
            with self.assertRaisesRegex(DatasetSchemaError, 'infinity'):
                self.mutated(config, lambda f: f.__setitem__(config.numeric_columns[0], np.inf))

    def test_wrong_student_subject_row_count(self):
        path = self.student.config.path.with_name('student-por.csv')
        with self.assertRaisesRegex(DatasetSchemaError, 'expected 33 columns; got 1'):
            load_dataset(self.student.config, path)
        with self.assertRaisesRegex(DatasetSchemaError, 'expected 395 rows, got 649'):
            load_dataset(replace(self.student.config, separator=','), path)

    def test_categorical_missing_and_unseen_in_pipeline(self):
        X = self.student.X.copy()
        X.loc[0, 'age'] = np.nan
        X.loc[1, 'Mjob'] = np.nan
        pipe = Pipeline([('preprocessor', make_preprocessor(self.student.config)), ('regressor', LinearRegression())])
        train = X.iloc[:300].copy()
        train['guardian'] = np.nan  # Entirely missing training column retains width.
        fitted = clone(pipe).fit(train, self.student.y.iloc[:300])
        test = X.iloc[300:].copy()
        test['Mjob'] = 'previously unseen occupation'
        self.assertTrue(np.isfinite(fitted.predict(test)).all())
        counts = feature_counts(self.student.config, fitted['preprocessor'])
        self.assertEqual(counts['original'], 30)
        self.assertGreater(counts['encoded'], 30)
        self.assertEqual(fitted['preprocessor'].transform(test).shape[1], counts['encoded'])

    def test_loader_retains_missing_predictors(self):
        def mutate(frame):
            frame.loc[0, 'age'] = np.nan
            frame.loc[1, 'Mjob'] = np.nan
        data = self.mutated(self.student.config, mutate)
        self.assertEqual(len(data.X), 395)
        self.assertTrue(data.X.loc[0:1].isna().any().any())

    def test_manufacturer_corrections_and_cardinality(self):
        from datasets.preprocessing import manufacturer_name
        expected = {'maxda rx3': 'mazda', 'Nissan rogue': 'nissan', 'porcshce panamera': 'porsche',
                    'toyouta corolla': 'toyota', 'vokswagen rabbit': 'volkswagen', 'vw golf': 'volkswagen'}
        for raw, brand in expected.items(): self.assertEqual(manufacturer_name(raw), brand)
        self.assertEqual(self.car.X.CarName.nunique(), 147)
        self.assertEqual(self.car.X.CarName.map(manufacturer_name).nunique(), 22)

    def test_car_unknown_categories_and_missing_values(self):
        from datasets.preprocessing import manufacturer_name
        self.assertTrue(pd.isna(manufacturer_name(None)))
        pre = make_preprocessor(CAR).fit(self.car.X.iloc[:160])
        test = self.car.X.iloc[160:].copy()
        test['CarName'] = 'unseenmanufacturer model'
        test.loc[test.index[0], 'horsepower'] = np.nan
        test.loc[test.index[1], 'fueltype'] = np.nan
        encoded = pre.transform(test)
        names = pre.get_feature_names_out()
        brand = [i for i,n in enumerate(names) if n.startswith('categorical__CarName_')]
        self.assertTrue(np.isfinite(encoded).all())
        np.testing.assert_array_equal(encoded[:, brand], 0)
        self.assertFalse(any('rx3' in n for n in names))

    def test_random_holdout_and_training_only_statistics(self):
        for data in (self.student, self.car):
            train, test = holdout_indices(data)
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(train)+len(test), len(data.y))
            self.assertTrue(data.temporal_metadata.empty)
            pre = make_preprocessor(data.config).fit(data.X.iloc[train])
            np.testing.assert_allclose(pre.named_transformers_['numeric']['imputer'].statistics_,
                                       data.X.iloc[train][list(data.config.numeric_columns)].median())
            self.assertGreater(feature_counts(data.config, pre)['encoded'], data.original_feature_count)
        self.assertEqual(tuple(map(len, holdout_indices(self.car))), (164, 41))

    def test_complete_pipeline_refits_preprocessing_inside_search(self):
        from unittest.mock import patch
        from sklearn.compose import ColumnTransformer
        from regression_experiment import make_pipeline
        data = self.car
        train, heldout = holdout_indices(data)
        X, y = data.X.iloc[train], data.y.iloc[train]
        folds = list(make_cv(CAR, 3).split(X))
        observed = []
        original = ColumnTransformer.fit_transform
        def record(transformer, X, *args, **kwargs):
            observed.append(frozenset(X.index))
            return original(transformer, X, *args, **kwargs)
        with patch.object(ColumnTransformer, 'fit_transform', record):
            search = GridSearchCV(make_pipeline(CAR, 'Correlation'), {'selector__k': [2, 4]},
                                  cv=folds, scoring='neg_root_mean_squared_error', error_score='raise')
            search.fit(X, y)
        expected = [frozenset(X.iloc[tr].index) for tr,va in folds] * 2 + [frozenset(X.index)]
        self.assertCountEqual(observed, expected)
        self.assertTrue(all(not (rows & set(heldout)) for rows in observed))

    def test_all_five_methods_accept_both_raw_datasets(self):
        from regression_experiment import make_pipeline, METHODS
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            for data in (self.student, self.car):
                train, test = holdout_indices(data)
                for method in METHODS:
                    fitted = make_pipeline(data.config, method, k=3).fit(data.X.iloc[train], data.y.iloc[train])
                    self.assertTrue(np.isfinite(fitted.predict(data.X.iloc[test])).all())
                    self.assertEqual(int(fitted['selector'].get_support().sum()), 3)

    def test_feature_name_order_matches_selector_transform(self):
        from regression_experiment import make_pipeline, selected_features
        data = self.car
        train, test = holdout_indices(data)
        fitted = make_pipeline(CAR, 'Correlation', k=8).fit(data.X.iloc[train], data.y.iloc[train])
        names = fitted['preprocessor'].get_feature_names_out()
        transformed = fitted['preprocessor'].transform(data.X.iloc[test])
        lookup = {n:i for i,n in enumerate(names)}
        manual = fitted['regressor'].intercept_ + transformed[:,[lookup[n] for n in selected_features(fitted)]] @ fitted['regressor'].coef_
        np.testing.assert_allclose(manual, fitted.predict(data.X.iloc[test]))

    def test_identifier_validation(self):
        with self.assertRaisesRegex(DatasetSchemaError, 'unique nonmissing'):
            self.mutated(CAR, lambda f: f.__setitem__('car_ID', 1))

    def test_random_split_reproducible_and_invalid_split(self):
        a = holdout_indices(self.student)
        b = holdout_indices(self.student)
        for x, y in zip(a, b): np.testing.assert_array_equal(x, y)
        for size in (0, 1, -0.2):
            with self.assertRaises(ValueError): holdout_indices(self.car, size)
        with self.assertRaises(ValueError): holdout_indices(self.car, gap=1)

if __name__ == '__main__':
    unittest.main()
