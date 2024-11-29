import os
import yaml
from joblib import dump, load
import numpy as np
import pandas as pd
from pygpg.sk import GPGRegressor
from pygpg.complexity import compute_complexity
from sklearn.model_selection import GridSearchCV
from modeling_helper_funcs import (
    create_model_performance_report,
    save_model,
    save_model_performance,
)
from modeling_helper_funcs import get_cv, extract_x_y, plot_grid_search


def generate_string_list(n):
    return [f"x_{i}" for i in range(n)]


def lists_to_dict(a, b):
    if len(a) != len(b):
        raise ValueError("Lists a and b must be of the same length.")
    return {a[i]: b[i] for i in range(len(a))}


def main():
    stage = "prepare_modelling"
    params = yaml.safe_load(open("/workspace/growthcurves/params.yaml"))
    paths = params["config"]["paths"]
    model_out_path = paths["models"] + "/symbolic/"
    os.chdir(paths["home"])
    os.makedirs(model_out_path, exist_ok=True)

    # set seed
    seed = params["seed"]
    np.random.seed(seed)
    datapoints_per_subject = params["train"]["datapoints_per_sub"]
    n_splits = params["train"]["n_splits"]
    test_size = params["train"]["test_size"]
    if params["train_symbolic"]["run_full_cv"]:
        evaluations = params["train_symbolic"]["evaluations"]
        tree_depth = params["train_symbolic"]["tree_depth"]
        # finetune_evals = params["train_symbolic"]["finetune_evals"]
        finetune_strategy = params["train_symbolic"]["finetune_strategy"]
        verbosity = params["train_symbolic"]["verbosity"]
        time_limit = params["train_symbolic"]["time_limit"]
        generations = params["train_symbolic"]["generations"]
        feat_sel = params["train_symbolic"]["feat_sel"]
    else:
        evaluations = [100]
        tree_depth = [2, 3]
        finetune_evals = [100]
        finetune_strategy = [True]
        verbosity = [True]
        time_limit = [1]
        generations = [1]

    model_data_path = os.path.join(paths["features"], "modelling_dataset.joblib")
    modelling_dataset = load(model_data_path)

    x_train = modelling_dataset["x_train"]
    y_train = modelling_dataset["y_train"]
    x_test = modelling_dataset["x_test"]
    y_test = modelling_dataset["y_test"]
    group_train = modelling_dataset["group_train"]

    x_colnames = x_train.columns
    y_colname = y_train.columns[0]

    # for the models that have hyperparameters we want to tune
    # create a custom cross validation scheme, optimize params over cv
    custom_cv = get_cv(datapoints_per_subject, seed, test_size, n_splits)
    parameters = {
        "e": evaluations,
        "d": tree_depth,
        #"finetune_max_evals": finetune_evals,
        "finetune": finetune_strategy,
        "verbose": verbosity,
        "t": time_limit,
        "g": generations,
        "feat_sel": feat_sel,
        "random_state": [seed]
    }
    cv_model = GPGRegressor(random_state=seed)
    grid_search = GridSearchCV(
        cv_model,
        parameters,
        cv=custom_cv,
        scoring="neg_mean_squared_error",
        verbose=True,
    )
    grid_search.fit(x_train, y_train, groups=group_train)

    # plot grid
    plot_grid_search(grid_search, parameters, "GPG", paths)

    model1 = GPGRegressor(**grid_search.best_params_)
    model1_fit = model1.fit(x_train, y_train.values.ravel())
    save_model(model1, "gpg", model_out_path)

    # get variable conversion dictionary
    # this is used to convert a string representation of the model with names like x_0, x_1 etc to a
    # model_str with correct (informative) variable name
    cols = x_colnames
    var_names = generate_string_list(len(cols))
    variable_dictionary = lists_to_dict(var_names, cols)
    model_str = str(model1.model.subs(variable_dictionary))

    # predict to give an initial result
    y_pred = model1.predict(x_test)
    model_performance_report = create_model_performance_report(
        x_train,
        x_test,
        y_train,
        y_test,
        y_pred,
        model1,
        grid_search.best_params_,
        "gpg",
        grid_search.best_params_,
        model_str,
    )
    save_model_performance(model_performance_report, "gpg", model_out_path)


if __name__ == "__main__":
    main()
