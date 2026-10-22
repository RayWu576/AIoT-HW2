"""Inspect official datasets without training or altering dashboard artifacts."""
import argparse
from . import load_student_performance, load_car_price, load_appliances_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-profile", default="early-30", choices=["early-30", "prior-grades-32"],
                        help="Original math CSV has 32 predictors; early-30 explicitly excludes G1/G2")
    parser.add_argument("--include-historical-car", action="store_true", help="Also inspect the historical Car Price dataset")
    args = parser.parse_args()
    datasets = [load_student_performance(include_prior_grades=args.student_profile == "prior-grades-32"), load_appliances_energy()]
    if args.include_historical_car:
        datasets.append(load_car_price())
    for data in datasets:
        print(f"{data.config.display_name}: rows={len(data.y)}, raw columns={len(data.raw_columns)}, original predictors={data.original_feature_count}, target={data.config.target} ({data.config.target_unit}), split={data.config.split_strategy}")
        print("Predictors:", ", ".join(data.X.columns))
        print("Numeric:", len(data.config.numeric_columns), "Categorical:", len(data.config.categorical_columns))
        if not data.temporal_metadata.empty:
            print("Time range:", data.temporal_metadata.date.iloc[0], "to", data.temporal_metadata.date.iloc[-1])

if __name__ == "__main__":
    main()
