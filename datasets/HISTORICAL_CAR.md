# Historical Car Price data workflow

This is not the current official Problem 2. Commands and split details below describe the previous phase only.

## Historical reference - Car Price Prediction

Kaggle: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction

Downloaded public archive: https://www.kaggle.com/api/v1/datasets/download/hellbuoy/car-price-prediction

Verified CarPrice_Assignment.csv: **205 observations, 26 columns**, no missing values in this copy. Exclude the identifier car_ID and target price from X, leaving **24 original candidate inputs**: 14 numerical and 10 categorical. Target unit: **USD**. Identifiers must be unique/nonmissing but are never predictors. Source URL, column list, row count and SHA-256 are recorded in car_price/source.json.

Actual complete CSV schema:

```text
car_ID, symboling, CarName, fueltype, aspiration, doornumber, carbody,
drivewheel, enginelocation, wheelbase, carlength, carwidth, carheight,
curbweight, enginetype, cylindernumber, enginesize, fuelsystem,
boreratio, stroke, compressionratio, horsepower, peakrpm, citympg,
highwaympg, price
```

Numeric inputs: symboling, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg.

Categorical inputs: CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem. Number words in door/cylinder fields remain nominal categories; no assumptions about equally spaced effects are imposed.

### CarName decision

CarName contains 147 distinct vehicle names among 205 observations. Encoding all names risks sparse model-specific indicators and overfitting. The Pipeline replaces its values with the first whitespace-separated manufacturer token, lowercases it, and applies fixed lexical corrections:

| Token | Corrected brand |
| --- | --- |
| maxda | mazda |
| porcshce | porsche |
| toyouta | toyota |
| vokswagen, vw | volkswagen |
| Nissan | nissan (lowercasing) |

This produces 22 brands in the complete authentic file. The mapping uses no target values or learned global frequencies. Brand imputation and vocabulary fitting still occur only inside each training fold. Unknown manufacturers receive all-zero one-hot indicators via handle_unknown="ignore". CarName remains one of the **24 original candidate fields**; manufacturer extraction does not inflate that count. The current 164-observation training set produces **71 encoded columns** in total; fold vocabularies and encoded counts can differ.

## Acquisition

Place files at student_performance/student-mat.csv and car_price/CarPrice_Assignment.csv under this directory. Both required files are included locally. To reacquire the car file with an authenticated Kaggle CLI:

```powershell
kaggle datasets download -d hellbuoy/car-price-prediction -p datasets/car_price --unzip
```

Or download the archive from the Kaggle page/public endpoint above. Obtain Mathematics from the UCI nested archive as documented; the specified student Kaggle copy supplies a different subject. If authentication or networking blocks acquisition, download through the source page with your own account and place the authentic CSV at the documented path. Never commit Kaggle credentials. Missing files produce a clear FileNotFoundError; wrong headers, duplicate headers, wrong row counts, invalid numerics/infinity or missing targets raise DatasetSchemaError. Missing predictors are retained for fold-local imputation.

## Pipeline and split contract

```python
from datasets import load_student_performance, load_car_price, make_preprocessor, holdout_indices, make_cv
student = load_student_performance()  # official 30 original inputs
car = load_car_price()               # 24 original candidate inputs
train, test = holdout_indices(car)   # seeded 164/41 random split
cv = make_cv(car.config, n_splits=4)  # apply only to raw training rows
preprocessor = make_preprocessor(car.config)  # unfitted ColumnTransformer
```

Numeric missing values use training-fold medians, followed by StandardScaler. Nominal features use training-fold mode imputation and OneHotEncoder(handle_unknown="ignore", sparse_output=False). Entirely missing training columns are retained. CarName manufacturer extraction is a cloneable sklearn transformer inside the categorical Pipeline. No preprocessing is fitted before GridSearchCV. Every candidate uses preprocessing - selector - LinearRegression; all learning occurs on that fold's training rows.

Both official datasets use reproducible random holdout/KFold rather than a temporal split. All index lists and random_state=0 are recorded under results/<dataset_id>/metadata.json. Original feature counts (30, 24) satisfy 14-39; feature_counts() reports encoded width separately after fitting. For method semantics, search bounds, actual execution and validation commands, see the repository README.
