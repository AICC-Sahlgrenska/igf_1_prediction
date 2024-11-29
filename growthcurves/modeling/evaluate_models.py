import os
import json
import joblib
import re
import collections
import pandas as pd
import yaml
import glob
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


def main():
    stage = "evaluate"
    params = yaml.safe_load(open("/workspace/growthcurves/params.yaml"))
    paths = params["config"]["paths"]
    os.chdir(paths["home"])
    os.makedirs(paths["results"], exist_ok=True)

    # Get data
    model_data_path = os.path.join(paths["features"], "modelling_dataset.joblib")
    modelling_dataset = joblib.load(model_data_path)

    x_test = modelling_dataset["x_test"]
    y_test = modelling_dataset["y_test"]

    results = collections.defaultdict(dict)
    all_models = glob.glob(paths["models"] + "/*/*" + ".joblib")

    pred_output_table = pd.concat([x_test, y_test], axis=1)
    
    for model_path in all_models:
        # Load pre-trained model
        model1 = joblib.load(model_path)
        model_name = re.split(r"[/.]", model_path)[2]

        y_pred = model1.predict(x_test)
        results[model_name]["r2"] = r2_score(y_test, y_pred)
        results[model_name]["mae"] = mean_absolute_error(y_test, y_pred)
        results[model_name]["mape"] = mean_absolute_percentage_error(y_test, y_pred)

        if 'ElasticNet' in model_name:
            results[model_name]["n_features"] = int(sum(model1.coef_ != 0))
        elif 'Explainable' in model_name:
            results[model_name]["n_features"] = int(len(model1.term_names_))
        elif 'gpg' in model_name:
            results[model_name]["n_features"] = int(len(model1.model.free_symbols))
        elif 'Dummy' in model_name:
            results[model_name]["n_features"] = int(0)
        else:
            results[model_name]["n_features"] = int(len(model1.feature_names_in_))

        
        pred_output_table[f'prediction_{model_name}'] = y_pred
    
    pred_output_table.to_csv(paths["results"] + "prediction_output_all_models.csv")

    with open(paths["results"] + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=4)
        
    results_df = pd.DataFrame.from_dict(results,  orient='index')
    results_df.to_csv(paths["results"] + "results_table.csv")


if __name__ == "__main__":
    main()
