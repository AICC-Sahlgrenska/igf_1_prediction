import os
import numpy as np
from joblib import dump
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy.stats as stats
from modeling_helper_funcs import (
    custom_scale,
    create_model_performance_report,
    save_model,
    save_model_performance,
    plot_grid_search,
)
from modeling_helper_funcs import get_cv, load_hold_out_data, load_train_data, extract_x_y


def main():
    stage = "train"
    params = yaml.safe_load(open("/workspace/growthcurves/params.yaml"))
    paths = params["config"]["paths"]
    model_out_path = paths["models"] + "/other/"
    os.chdir(paths["home"])
    os.makedirs(model_out_path, exist_ok=True)

    # Parameters
    seed = params["seed"]
    datapoints_per_subject = params["train"]["datapoints_per_sub"]
    n_splits = params["train"]["n_splits"]
    test_size = params["train"]["test_size"]
    x_colnames = params["train"]["x_colnames"]
    y_colname = params["train"]["y_colname"]

    # get data & format
    df = load_train_data(paths)
    hold_out_df = load_hold_out_data(paths)

    # add in all variables that are called something "_delta_"
    # this is not done manually in x_colnames list in params
    # to make it easier to change between different types of delta (up/down vs continuous)
    delta_colnames = [x for x in df.columns if "_delta_" in x]
    x_colnames = x_colnames + delta_colnames

    model_factories = {
        "DummyRegressor": lambda **kwargs: DummyRegressor(**kwargs),
        "LinearRegression": lambda **kwargs: LinearRegression(**kwargs),
        "ElasticNet": lambda **kwargs: ElasticNet(**kwargs),
        "ElasticNetCV": lambda **kwargs: ElasticNetCV(**kwargs),
        "ExplainableBoostingRegressor": lambda **kwargs: ExplainableBoostingRegressor(
            **kwargs
        ),
    }

    cv_params = {
        "DummyRegressor": False,
        "LinearRegression": False,
        "ElasticNet": {
            "alpha": np.logspace(-4, 0.5, 20),
            "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
            "selection": ["random"],
        },
        "ElasticNetCV": False,
        # "ExplainableBoostingRegressor": {
        #     "min_samples_leaf": [2, 3, 4],
        #     'learning_rate': [0.01, 0.001, 0.2],
        #     "cyclic_progress": [0.6, 0.8, 1],
        #     'objective' : ['rmse'],
        #     'smoothing_rounds': [50, 200, 500, 1000],
        #     'greedy_ratio':  [0.0, 0.5, 1.0, 1.5, 2.0, 4.0],
        #     'validation_size': [0.1, 0.15, 0.2],
        #     'random_state': [seed]
        # },
        "ExplainableBoostingRegressor": {
            "min_samples_leaf": stats.randint(2, 4),
            'learning_rate': stats.uniform(0.001, 0.5),
            "cyclic_progress": stats.uniform(0, 1),
            'smoothing_rounds': stats.randint(100, 500),
            'interactions': stats.uniform(0, 1),
            'greedy_ratio': stats.uniform(0, 4),
            'random_state': [seed]
        }    
    }

    x_train, y_train, group_train = extract_x_y(df, x_colnames, y_colname)
    x_test, y_test, group_test = extract_x_y(hold_out_df, x_colnames, y_colname)

    print(f"Training on {len(y_train)} datapoints, Testing on {len(y_test)} datapoints")

    # apply scaler if requested
    if params[stage]["use_scaler"]:
        x_train, x_test, scaler_sds = custom_scale(x_train, x_test)
        print("done!")

    # Save training and test sets
    modelling_dataset = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "group_train": group_train,
        "group_test": group_test
    }

    output_file = os.path.join(paths["features"], "modelling_dataset.joblib")
    dump(modelling_dataset, output_file)

    for model_name, factory in model_factories.items():
        model_params = {}
        if cv_params[model_name] and params[stage]["run_full_cv"]:
            # for the models that have hyperparameters we want to tune
            # create a custom cross validation scheme, optimize params over cv
            custom_cv = get_cv(datapoints_per_subject, seed, test_size, n_splits)
            cv_model = factory(**model_params)
            param_grid = cv_params[model_name]
            X, y, groups = extract_x_y(df, x_colnames, y_colname)
            if model_name=="ExplainableBoostingRegressor":
                grid_search = RandomizedSearchCV(
                    cv_model,
                    param_distributions = param_grid,
                    n_iter=100,
                    cv=custom_cv,
                    scoring="neg_root_mean_squared_error",
                    verbose=True,
                )
                grid_search.fit(X, y, groups=groups)
            else:
                grid_search = GridSearchCV(
                    cv_model,
                    param_grid,
                    cv=custom_cv,
                    scoring="neg_root_mean_squared_error",
                    verbose=True,
                )
                grid_search.fit(X, y, groups=groups)

            # plot grid
            plot_grid_search(grid_search, param_grid, model_name, paths)

            # update the model_params dictionary with all optimized variables
            for key in cv_params[model_name]:
                optimal_param = grid_search.best_params_[key]
                model_params[key] = optimal_param

        if "CV" in model_name:
            model_params = {"cv": params[stage]["n_splits"]}
        if "ExplainableBoostingRegressor" in model_name:
            model_params["n_jobs"] = 1
        # create a model call based on model_params
        # for non-cv models the model_params will be empty
        # and for models where we ran cv model_params will contain the optimal parameters according to cv
        model = factory(**model_params)
        model = model.fit(x_train, y_train.values.ravel())
        save_model(model, model_name, model_out_path)

        y_pred = model.predict(x_test)
        model_performance_report = create_model_performance_report(
            x_train, x_test, y_train, y_test, y_pred, model, model_params, model_name, cv_params
        )
        save_model_performance(model_performance_report, model_name, model_out_path)


if __name__ == "__main__":
    main()
