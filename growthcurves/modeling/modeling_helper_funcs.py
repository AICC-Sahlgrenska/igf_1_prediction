import os
import pandas as pd
import numpy as np
import datetime
from joblib import dump
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns


class GroupShuffleNDataPoints:
    # custom cross validation class to enable both stratification by patient
    # and selecting a maximum of N points per subject
    # (with different points per fold)
    def __init__(
        self, n_splits=5, test_size=None, train_size=None, random_state=None, n_points=2
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.n_points = n_points  # Number of data points per patient for test set

    def split(self, X, y, groups):
        cv_generator = GroupShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        )

        rng = check_random_state(self.random_state)

        for train_idx, test_idx in cv_generator.split(X, y, groups):
            unique_groups = np.unique(groups)
            train_idx_reduced = []
            test_idx_reduced = []

            for group in unique_groups:
                group_indices = np.where(groups == group)[0]
                # Separately reduce both training and test groups
                group_train_idx = np.intersect1d(group_indices, train_idx)
                group_test_idx = np.intersect1d(group_indices, test_idx)

                if len(group_train_idx) > self.n_points:
                    group_train_idx = rng.choice(
                        group_train_idx, size=self.n_points, replace=False
                    )

                if len(group_test_idx) > self.n_points:
                    group_test_idx = rng.choice(
                        group_test_idx, size=self.n_points, replace=False
                    )

                train_idx_reduced.extend(group_train_idx)
                test_idx_reduced.extend(group_test_idx)

            # Now yield the indices with limited data points per group for both train and test sets
            yield np.array(train_idx_reduced), np.array(test_idx_reduced)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def load_train_data(paths):
    df = pd.read_csv(os.path.join(paths["features"], "train.csv"))
    return df


def load_hold_out_data(paths):
    df = pd.read_csv(os.path.join(paths["features"], "hold-out-validation.csv"))
    return df


def split_test_train(data, seed):
    subjects = data.patno.unique()
    subjects_train, subjects_test = train_test_split(
        subjects, test_size=0.2, random_state=seed
    )
    train_df = data[data.patno.isin(subjects_train)]
    test_df = data[data.patno.isin(subjects_test)]
    return train_df, test_df


def extract_x_y(data, x_colnames, y_colname):
    # TODO: lift colnames to params
    y = data[y_colname]
    group = data["patno"]
    x = data[x_colnames]
    return x, y, group


def get_cv(datapoints_per_subject, seed, test_size, n_splits):
    # get cross validation splits
    # either for all data points or a subset per subject (depending on datapoints_per_subject)
    if datapoints_per_subject == "all":
        # if using all data points per sub, cv by GroupShuffleSplit
        cv = GroupShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=seed
        )
    elif isinstance(datapoints_per_subject, int) and (datapoints_per_subject > 0):
        # if limiting to N datapoints per subject, use custom cv class
        cv = GroupShuffleNDataPoints(
            n_splits=n_splits,
            test_size=test_size,
            random_state=seed,
            n_points=datapoints_per_subject,
        )
    else:
        raise Exception(
            f"param datapoints_per_sub should be a positive integer or 'all', you had {datapoints_per_subject}"
        )

    return cv


def custom_scale(x_train, x_test):
    numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    scaler.fit(x_train[numeric_features])
    x_train_scaled = scaler.transform(x_train[numeric_features])
    x_test_scaled = scaler.transform(x_test[numeric_features])
    x_train.loc[:, numeric_features] = pd.DataFrame(
        x_train_scaled, index=x_train.index, columns=numeric_features
    )
    x_test.loc[:, numeric_features] = pd.DataFrame(
        x_test_scaled, index=x_test.index, columns=numeric_features
    )
    feature_stds = scaler.scale_
    return x_train, x_test, feature_stds


def adjusted_r2(r2, x):
    n = len(x)  # sample size
    k = len(x.columns)  # number of parameters in the model
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))


def get_standardised_coefficient_linear_models(variable, coef, x_train, y_train):
    standardised_coef = coef * (np.std(x_train[variable])/np.std(y_train.iloc[:,0]))
    return standardised_coef


def add_predictive_variables_regression(model, x_train, y_train, model_performance_report):
    model_performance_report += "Coef;Standardised coef;Feature name\n"
    columns_zero_coef = []
    coef_lines = []
    for coef, feature in sorted(zip(model.coef_, x_train.columns), reverse=True):
        standardised_coef = get_standardised_coefficient_linear_models(feature, coef, x_train, y_train)
        formatted_line = f"{str(coef)};{str(standardised_coef)};{feature}\n"
        if coef == 0:
            columns_zero_coef.append(formatted_line)
            continue
        coef_lines.append(formatted_line)
    model_performance_report += "".join(coef_lines)
    if len(columns_zero_coef) > 0:
        model_performance_report += "Coef zero;standardised coef;Feature\n"
        model_performance_report += "".join(columns_zero_coef)
    return model_performance_report


def add_predictive_variables_ebm(model, model_performance_report):
    ebm_global = model.explain_global()
    ebm_global.data()
    feature_names = ebm_global.data()["names"]
    feature_scores = ebm_global.data()["scores"]
    max_length_vector2 = max(len(str(score)) for score in feature_scores)
    model_performance_report += "{:<{width}}  {}\n".format(
        "Importance", "Feature", width=max_length_vector2 + 2
    )
    for score, name in sorted(zip(feature_scores, feature_names), reverse=True):
        model_performance_report += "{:<{width}}  {}\n".format(
            str(score), name, width=max_length_vector2 + 2
        )
    return model_performance_report



def create_model_performance_report(
    x_train,
    x_test,
    y_train,
    y_test,
    y_pred,
    model,
    model_params,
    model_name,
    cv_params,
    model_string="",
):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_r2 = r2_score(y_test, y_pred)
    model_r2_adj = adjusted_r2(model_r2, x_test)
    model_mae = mean_absolute_error(y_test, y_pred)
    model_mape = mean_absolute_percentage_error(y_test, y_pred)
    model_performance_report = (
        f"Model performance {model_name}\n"
        f"Timestamp: {current_time}\n"
        f"R2: {model_r2}\n"
        f"Adjusted R2: {model_r2_adj}\n"
        f"Mean Absolute Error: {model_mae}\n"
        f"Mean Absolute Percentage Error: {model_mape}\n"
    )
    model_performance_report += model_string + "\n"
    if cv_params:
        for model_param in model_params.keys():
            model_performance_report += (
                f"{model_param}: {model_params[model_param]}{os.linesep}"
            )
        if "ElasticNet" in model_name or "LinearRegression" in model_name:
            if "CV" in model_name:
                # manually pick out the best Alpha and L1 ratio from ElasticNetCV
                model_performance_report += f"Alpha: {model.alpha_}"
                model_performance_report += (
                    f"{os.linesep}L1 Ratio: {model.l1_ratio_}{os.linesep}"
                )
            model_performance_report = add_predictive_variables_regression(
                model, x_train, y_train, model_performance_report
            )
        if "ExplainableBoosting" in model_name:
            model_performance_report = add_predictive_variables_ebm(
                model, model_performance_report
            )
    return model_performance_report


def safe_sample(group_df, datapoints, random_seed):
    n_samples = min(len(group_df), datapoints)
    return group_df.sample(n=n_samples, random_state=random_seed)


def save_model(model, model_type, path):
    output_file = os.path.join(path, model_type + "_model.joblib")
    dump(model, output_file)


def save_model_performance(model_performance_report, model_type, path):
    with open(path + "/" + model_type + "_model_performance.txt", "a") as text_file:
        text_file.write(model_performance_report)


def plot_grid_search(grid_search, param_grid, model_name, paths):
    cv_results = grid_search.cv_results_
    mean_test_scores = cv_results["mean_test_score"]
    param_names = list(param_grid.keys())
    param_vals = [param_grid[name] for name in param_names]
    for param1, param2 in combinations(param_names, 2):
        scores = []
        p1_values = sorted(set(cv_results[f"param_{param1}"]))
        p2_values = sorted(set(cv_results[f"param_{param2}"]))
        if (len(p1_values) < 2) or (len(p2_values)) < 2:
            # don't make plots where one dimension is just one value
            continue
        else:
            for p1 in p1_values:
                row = []
                for p2 in p2_values:
                    mask = (cv_results[f"param_{param1}"] == p1) & (
                        cv_results[f"param_{param2}"] == p2
                    )
                    score = mean_test_scores[mask]
                    if len(score) > 0:
                        row.append(score[0])
                    else:
                        row.append(np.nan)
                scores.append(row)
            scores = pd.DataFrame(scores, index=p1_values, columns=p2_values)

            plt.figure(figsize=(10, 8))
            sns.heatmap(scores, annot=True, fmt=".3f", cmap="viridis")
            plt.title(f"{model_name} - Mean Test Scores\n({param1} vs {param2})")
            plt.xlabel(param2)
            plt.ylabel(param1)
            plt.savefig(
                f"{paths['results']}/grid_search_{model_name}_{param1}_{param2}.png"
            )
