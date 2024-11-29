import yaml
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tableone import TableOne
from scipy import stats
from statsmodels.stats.multitest import multipletests
import seaborn as sns


pd.options.mode.chained_assignment = None  # default='warn'


def plot_nas(df: pd.DataFrame, cols: list):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100
        na_df = na_df[cols]
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({"Missing Ratio %": na_df})
        missing_data.plot(kind="barh")
        plt.savefig(
            "/workspace/growthcurves/reports/figures/data_coverage.png",
            bbox_inches="tight",
        )
    else:
        print("No NAs found")


def get_table1(
    df, stable_columns, variable_columns, name, pval_adj=None, nonnormal_cols=[]
):
    # write out table1 for the given data

    datapoints_per_sub = (
        df.groupby("patno").size().reset_index(name="n_datapoints_per_subject")
    )
    df = df.merge(datapoints_per_sub, on="patno", how="left", validate="many_to_one")

    check_normality = TableOne(
        df.groupby("patno").first().reset_index(),
        columns=stable_columns + ["n_datapoints_per_subject"] + variable_columns,
        groupby="group",
        pval=True,
        normal_test=True,
        dip_test=True,
        tukey_test=True,
    )

    # get nonnormal vars into list
    nonnormal_mask = ((check_normality.cont_describe.normality >= 0) &
                (check_normality.cont_describe.normality <= 0.001))
    nonnormal_vars = list(check_normality.cont_describe.normality[nonnormal_mask].
                    dropna(how='all').index)
    dip_mask = ((check_normality.cont_describe.hartigan_dip >= 0) &
                            (check_normality.cont_describe.hartigan_dip <= 0.05))
    dip_vars = list(check_normality.cont_describe.hartigan_dip[dip_mask].
                                dropna(how='all').index)
    outlier_mask = check_normality.cont_describe.far_outliers > 1
    outlier_vars = list(check_normality.cont_describe.far_outliers[outlier_mask].
                        dropna(how='all').index)
    
    modal_vars = nonnormal_vars + dip_vars + outlier_vars
    
    # for data whose value stays the same for all data points per subject (like age of starting gh)
    stable_table = TableOne(
        df.groupby("patno").first().reset_index(),
        columns=stable_columns + ["n_datapoints_per_subject"],
        nonnormal = modal_vars,
        min_max = modal_vars,
        groupby="group",
        pval=True,
        pval_adjust=pval_adj,
        dip_test=True,
        normal_test=True,
        tukey_test=True,
        htest_name = True
    )

    # detect our categorical variables
    positive_categorical = [x for x in df.columns if '_up' in x]
    negative_categorical = [x for x in df.columns if '_down' in x]
    categorical_vars = positive_categorical + negative_categorical
        
    # add higher precision to the columns that have very small values
    high_precision_cols = ['gh_dose_proportional_delta_3_m','gh_dose_proportional']
    high_precision_dict = dict(zip(high_precision_cols, [3]*len(high_precision_cols)))
    
    # for data whose value changes for each data points (like latest igf-1 value)
    variable_table = TableOne(
        df,
        columns=variable_columns,
        nonnormal = modal_vars,
        min_max = modal_vars,
        categorical = categorical_vars,
        groupby="group",
        pval=True,
        pval_adjust=pval_adj,
        dip_test=True,
        normal_test=True,
        tukey_test=True,
        decimals=high_precision_dict,
        htest_name = True
    )    

    outfilename = f"reports/table1/{name}_{pval_adj}_correction"

    stable_table.to_csv(outfilename + "_stable_cols.csv")
    variable_table.to_csv(outfilename + "_variable_cols.csv")

    stable_csv = pd.read_csv(outfilename + "_stable_cols.csv", skiprows=1)
    stable_csv.iloc[0, 0] = "n_subjects"

    variable_csv = pd.read_csv(outfilename + "_variable_cols.csv", skiprows=1)
    variable_csv.iloc[0, 0] = "n_datapoints"

    total_table1 = pd.concat(
        [
            stable_csv.iloc[[0]],
            variable_csv.iloc[[0]],
            stable_csv.iloc[[-1]],
            stable_csv[1:-1],
            variable_csv[1:],
        ],
        ignore_index=True,
    )
    
    # holm-correct p-values
    pvals = list(total_table1.loc[:, "P-Value"])
    # for the occasional < X value, fix that
    pvals_fixed = [float(pval[1:])/2 if (type(pval) == str and "<" in pval)  else float(pval) for pval in pvals ]

    corrections = multipletests(pvals_fixed, method = 'holm')[1]
    # move p-value for chi squared text in one hot encoded values value 0 to value 1
    corrections = pd.DataFrame(corrections).ffill()
    total_table1["P-value (Holm-corrected)"] = corrections

    # fix issues with two leading cols
    total_table1 = total_table1.rename(columns={'Unnamed: 1':'category_val'})
    total_table1.category_val = total_table1.category_val.fillna(1)
    total_table1.Missing = total_table1.Missing.fillna(0)    
    
    # carry over hypothesis test info to all categories
    total_table1['Test'] = total_table1['Test'].ffill()
    
    # limit output to categorical columns with true outcome
    total_table1_output = total_table1.query("category_val == 1")
    total_table1_output = total_table1_output.drop(columns=['category_val'], inplace=False)
    
    
    # write out
    total_table1_output.to_csv(outfilename + "_all.csv", index=False, sep=';')



def identify_nonnormal_columns(df, columns, stable_cols):
    nonnormal_cols = []
    for column in columns:
        # making sure we don't count stable cols multiple times
        if column in stable_cols:
            col = df.groupby('patno').first()[column]
        else: 
            col = df[column]
        stat, p_value = stats.normaltest(col, nan_policy='omit')  # Shapiro-Wilk-test
        if p_value < 0.05:
            nonnormal_cols.append(column)
    return nonnormal_cols


def main():
    stage = "prepare_modelling"
    params = yaml.safe_load(open("/workspace/growthcurves/params.yaml"))
    paths = params["config"]["paths"]
    os.makedirs(paths["table1"], exist_ok=True)
    os.chdir(paths["home"])

    # apply exclusion criteria
    required_columns = params[stage]["required_columns"]
    phases = params[stage]["acceptable_phases"]
    sexes = params[stage]["acceptable_sex"]

    data = pd.read_csv(os.path.join(paths["features"], "processed.csv"))
    
    data = data.dropna(subset=required_columns)
    data = data.query("phase in @phases")

    ## define colnames used for modeling & reporting
    x_colnames = params["train"]["x_colnames"]
    y_colname = params["train"]["y_colname"]
    delta_colnames = [x for x in data.columns if "_delta_" in x]
    all_colnames = x_colnames + delta_colnames + y_colname
    # columns that are always the same for all datapoints from a particular subject
    hist_colnames = [
        "birth_weight",
        "birth_length",
        "age_gh_start",
        # "gh_max_spontaneous",
        "gh_max_stimulation",
        "height_velocity_gh_start_1",
        "height_velocity_0",
        "height_velocity_1",
        "perc_change_igf1_gh_dos_date_3m",
        "perc_change_igf1_gh_dos_date_1y",
    ]
    active_colnames = [x for x in all_colnames if x not in hist_colnames]

    # after discussion on 25/5 2024 we decided to limit the primary modeling to boys
    # taking out data for the other sex separately (most likely girls, defined in params)
    data_othersex = data.query("sex not in @sexes")
    data_othersex.to_csv(
        os.path.join(paths["features"], "hold-out-other-sex.csv"), index=False
    )
    # limiting data to the relevant sex (most likely boys, defined in params )
    data = data.query("sex in @sexes")

    x_colnames = params["train"]["x_colnames"]
    delta_colnames = [x for x in data.columns if "_delta_" in x]
    x_colnames = x_colnames + delta_colnames
    plot_nas(data, x_colnames)

    corr_data = data.groupby('patno').first()
    corr_coefficient = corr_data['gh_max_spontaneous'].corr(corr_data['gh_max_stimulation'])
    sns.relplot(data=corr_data, x="gh_max_spontaneous", y="gh_max_stimulation")
    plt.text(x=0, y=21, s=f"Pearson's correlation coefficient: {round(corr_coefficient,2)}", fontsize=12, color='black')
    plt.savefig("/workspace/growthcurves/reports/figures/correlation_gh_max.png",
                bbox_inches="tight")

    # split to train and hold-out validation set by subject id
    subjects = data.patno.unique()
    subjects_train, subjects_test = train_test_split(
        subjects, test_size=params[stage]["hold_out_size"], random_state=params["seed"]
    )

    # TODO: can we do more imputations to increase the dataset?
    train_df = data[data.patno.isin(subjects_train)].dropna(subset=all_colnames)
    test_df = data[data.patno.isin(subjects_test)].dropna(subset=all_colnames)

    # create table 1 to show stats per group
    data["group"] = "train"
    data["group"] = data["group"].where(
        data["patno"].isin(subjects_train), other="hold-out"
    )

    stable_cols = hist_colnames + ["n_datapoints_per_subject"]
    # identify nonnormal columns
    hist_nonnormal_cols = identify_nonnormal_columns(data, hist_colnames, stable_cols)
    active_nonnormal_cols = identify_nonnormal_columns(data, active_colnames, stable_cols)
    all_nonnormal_cols = hist_nonnormal_cols + active_nonnormal_cols

    ## TODO: should we show non-imputed or imputed values?
    # with missing values
    get_table1(
        data,
        hist_colnames,
        active_colnames,
        "na_keep",
        pval_adj=None,
        nonnormal_cols=all_nonnormal_cols,
    )

    # drop all missing
    data_tmp = data[["group", "patno"] + all_colnames]
    data_tmp = data_tmp.dropna()
    get_table1(
        data_tmp,
        hist_colnames,
        active_colnames,
        "na_removed",
        pval_adj=None,
        nonnormal_cols=all_nonnormal_cols,
    )

    # save final result
    train_df.to_csv(os.path.join(paths["features"], "train.csv"), index=False)
    test_df.to_csv(
        os.path.join(paths["features"], "hold-out-validation.csv"), index=False
    )


if __name__ == "__main__":
    main()
