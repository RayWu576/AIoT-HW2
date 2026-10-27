"""Single source of truth for the official regression problems."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    display_name: str
    filename: str
    target: str
    target_unit: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    raw_columns: tuple[str, ...]
    expected_rows: int
    separator: str
    split_strategy: str
    excluded_columns: tuple[str, ...] = ()

    @property
    def feature_names(self):
        excluded = {*self.excluded_columns, self.target}
        return tuple(c for c in self.raw_columns if c not in excluded)

    @property
    def original_feature_count(self):
        return len(self.feature_names)

    @property
    def path(self):
        return ROOT / self.dataset_id / self.filename

STUDENT_COLUMNS = tuple("school sex age address famsize Pstatus Medu Fedu Mjob Fjob reason guardian traveltime studytime failures schoolsup famsup paid activities nursery higher internet romantic famrel freetime goout Dalc Walc health absences G1 G2 G3".split())
STUDENT_CATEGORICAL = tuple("school sex address famsize Pstatus Mjob Fjob reason guardian schoolsup famsup paid activities nursery higher internet romantic".split())

def student_config(*, include_prior_grades: bool):
    # Both profiles are explicit: the authentic CSV has 32 predictors, not 30.
    excluded = () if include_prior_grades else ("G1", "G2")
    numeric = tuple(c for c in STUDENT_COLUMNS if c not in (*STUDENT_CATEGORICAL, "G3", *excluded))
    return DatasetConfig("student_performance", "Student Performance (Mathematics)",
                         "student-mat.csv", "G3", "grade (0-20)", numeric,
                         STUDENT_CATEGORICAL, STUDENT_COLUMNS, 395, ";", "random", excluded)

CAR_COLUMNS = tuple("car_ID symboling CarName fueltype aspiration doornumber carbody drivewheel enginelocation wheelbase carlength carwidth carheight curbweight enginetype cylindernumber enginesize fuelsystem boreratio stroke compressionratio horsepower peakrpm citympg highwaympg price".split())
CAR_CATEGORICAL = tuple("CarName fueltype aspiration doornumber carbody drivewheel enginelocation enginetype cylindernumber fuelsystem".split())
CAR = DatasetConfig("car_price", "Car Price Prediction", "CarPrice_Assignment.csv", "price", "USD",
                    tuple(c for c in CAR_COLUMNS if c not in (*CAR_CATEGORICAL, "car_ID", "price")),
                    CAR_CATEGORICAL, CAR_COLUMNS, 205, ",", "random", ("car_ID",))
STUDENT = student_config(include_prior_grades=False)
LEGACY_DATASETS = {c.dataset_id: c for c in (STUDENT, CAR)}

# The original Phase 2 pair remains available as a historical scope.
ENERGY_COLUMNS = tuple(["date", "Appliances", "lights"] + [c for i in range(1, 10) for c in (f"T{i}", f"RH_{i}")] + ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint", "rv1", "rv2"])
ENERGY = DatasetConfig("appliances_energy", "Appliances Energy Prediction", "energydata_complete.csv",
                       "Appliances", "Wh",
                       tuple(c for c in ENERGY_COLUMNS if c not in ("date", "Appliances", "rv1", "rv2")),
                       (), ENERGY_COLUMNS, 19735, ",", "chronological", ("date", "rv1", "rv2"))
PHASE2_DATASETS = {c.dataset_id: c for c in (STUDENT, ENERGY)}

# Official problems are ordered for the dashboard: Student, Energy, Car.
OFFICIAL_DATASETS = {c.dataset_id: c for c in (STUDENT, ENERGY, CAR)}
