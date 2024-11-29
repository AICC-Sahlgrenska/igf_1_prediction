import yaml
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from pandas.tseries.offsets import DateOffset

pd.options.mode.chained_assignment = None  # default='warn'


def convert_dates(df):
    # ensure all date columns are in datetime format
    date_columns = [col for col in df.columns if "date" in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col], errors="coerce"
        )  # 'coerce' will set invalid parsing to NaT
    return df


def find_puberty_start_date(df):
    # find puberty start date for boys
    mask_testis_l = df["testis_l_ml"] >= 4
    mask_testis_r = df["testis_r_ml"] >= 4

    # Use boolean indexing to filter rows, then group by 'tiger_id' and get the first occurrence
    left_testis_puberty = (
        df[mask_testis_l]
        .groupby("tiger_id")
        .first()
        .reset_index()[["tiger_id", "visit_date"]]
        .rename(columns={"visit_date": "puberty_left_testis"})
    )
    right_testis_puberty = (
        df[mask_testis_r]
        .groupby("tiger_id")
        .first()
        .reset_index()[["tiger_id", "visit_date"]]
        .rename(columns={"visit_date": "puberty_right_testis"})
    )
    # combine the two
    testis_puberty = pd.merge(left_testis_puberty, right_testis_puberty)
    # pick the earlier date
    testis_puberty["puberty_start_date"] = testis_puberty[
        ["puberty_right_testis", "puberty_left_testis"]
    ].min(axis=1)

    # find puberty start for girls
    mask_breast = (df["breast_&_genitalia"] >= 2) & (df["sex"] == "Female")
    breast_puberty = (
        df[mask_breast]
        .groupby("tiger_id")
        .first()
        .reset_index()[["tiger_id", "visit_date"]]
        .rename(columns={"visit_date": "puberty_start_date"})
    )

    puberty_start = pd.concat(
        [testis_puberty[["tiger_id", "puberty_start_date"]], breast_puberty]
    )

    return puberty_start


def fix_puberty_start_date(data, corrections):
    df = data.copy()
    for index, row in corrections.iterrows():
        curr_sub = row["tiger_id"]
        df.loc[df["tiger_id"] == curr_sub, "puberty_start_date"] = row["pubertetstart"]
    return df


# The function to find the closest value to each integer within a group
def find_closest_age(group, targets, value_column):
    closest_values = {}
    for target in targets:
        # Compute the absolute difference with target and identify the minimum
        idx = (group["age"] - target).abs().idxmin()
        closest_values[target] = group.loc[idx, value_column]
    return pd.Series(closest_values)


def find_closest_date(
    group, targets, value_column, n_years=0, n_months=0, direction="backwards"
):
    closest_values = {}
    for target in targets:
        if direction == "forwards":
            # find visit a y_years, n_months forward in time
            idx_year = (
                (
                    group["visit_date"]
                    - (group[target] + pd.DateOffset(years=n_years, months=n_months))
                )
                .abs()
                .idxmin()
            )
            closest_values[f"{target}+{n_years}y_{n_months}m"] = group.loc[
                idx_year, value_column
            ]
        else:
            # find visit a y_years, n_months before
            idx_year = (
                (
                    group["visit_date"]
                    - (group[target] - pd.DateOffset(years=n_years, months=n_months))
                )
                .abs()
                .idxmin()
            )
            closest_values[f"{target}-{n_years}y_{n_months}m"] = group.loc[
                idx_year, value_column
            ]
        # Compute the absolute difference with target and identify the minimum
        idx = (group["visit_date"] - group[target]).abs().idxmin()
        closest_values[target] = group.loc[idx, value_column]

    return pd.Series(closest_values)


def determine_treatment_phase(row):
    if row["visit_date"] <= row["gh_start_date"]:
        return "pre-treatment"
    elif row["gh_start_date"] <= row["visit_date"] < row["gh_dos_m_date"]:
        return "catch-up"
    elif row["gh_dos_m_date"] <= row["visit_date"] < row["gh_stop_date"]:
        return "maintenance"
    elif row["gh_stop_date"] <= row["visit_date"]:
        return "post-treatment"
    else:
        # most commonly the error comes from a missing gh_stop_date
        return "ERROR"


def categorize_change(value, threshold):
    if value > threshold:
        return "up"
    elif value < -threshold:
        return "down"
    else:
        return "avg same"


def get_values_delta_months(
    df,
    column,
    months,
    merge_tolerance_days=30,
    percentage=False,
    only_direction=False,
    direction_change_limit=10,
    use_imputed = True
):
    if use_imputed and f"{column}_imputed" in df.columns:
        # for each date, find a date ~N months back
        df_tmp = column_time_delta(df, f"{column}_imputed", months)
        df_tmp = df_tmp.rename(columns={f"{column}_imputed_{months}_m_ago": f"{column}_{months}_m_ago"})
    else:
        # for each date, find a date ~N months back
        df_tmp = column_time_delta(df, column, months)
    df = df.sort_values(["visit_date"])
    df_tmp = df_tmp.sort_values(["shifted_visit_date"])
    new_column = f"{column}_delta_{months}_m"

    # merge
    df_out = pd.merge_asof(
        df,
        df_tmp,
        by=["tiger_id", "patno"],
        left_on="visit_date",
        right_on="shifted_visit_date",
        tolerance=pd.Timedelta(days=merge_tolerance_days),  # set the tolerance for matching
        direction="nearest",  # find the nearest date, either before or after the key
    )
    df_out = df_out.drop(columns={"shifted_visit_date"})
    
    # fix output to correct format
    df_out[new_column] = df_out[column] - df_out[f"{column}_{months}_m_ago"]
    # determine how the delta should be presented - in original units, as percentage or just directional
    if percentage or only_direction:
        # if necessary, calculate % change from x months ago
        df_out[new_column] = (
            df_out[new_column] / df_out[f"{column}_{months}_m_ago"]
        ) * 100
        # TODO: Fix this to be row-wise
        if only_direction:
            df_out[new_column] = df_out[new_column].apply(categorize_change, threshold=direction_change_limit)
            df_out = pd.get_dummies(df_out, prefix=new_column, columns=[new_column], drop_first=True)
    return df_out


def column_time_delta(df, column, shift=6):
    df = df.copy()[["patno", "tiger_id", "visit_date", column]]
    df["shift_by_months"] = shift
    df["shifted_visit_date"] = df["visit_date"] + DateOffset(months=shift)
    df = df.dropna(subset=column)
    df = df.rename(columns={column: f"{column}_{shift}_m_ago"})
    df = df.drop(columns={"visit_date", "shift_by_months"})
    return df


def igf1_to_target(df, target_column='igf_1', shift=6):
    df = df[["tiger_id", "visit_date", target_column]]
    #df = df.drop(columns=["igf_1_sds", "igfbp_3_sds", "igf_1_igfbp_3_sds"])
    df["shift_by_months"] = shift
    # the date the target value is measured is called "target_visit_date"
    df = df.rename(columns={"visit_date": "target_visit_date"})
    # the date when the prediction visit happens is called "visit_date"
    df["visit_date"] = pd.to_datetime(df["target_visit_date"]) - pd.DateOffset(
        months=shift
    )
    df = df.rename(
        columns={
            target_column: f"target_{target_column}",
        }
    )
    df = df.drop(columns={"shift_by_months"})
    return df


def combine_data_with_target(df, df_target, merge_tolerance=30):

    # prep data for merge of predictors and targets
    df = df.rename(columns={"patno_x": "patno"})
    df = df.sort_values(["visit_date"])
    df_target = df_target.sort_values(["visit_date"])
    # df_target = df_target.drop(columns='visit_date')

    df_target.visit_date = df_target.visit_date.astype("datetime64[ns]")
    # add target
    df_out = pd.merge_asof(
        df,
        df_target,
        by=["tiger_id"],
        left_on="visit_date",
        right_on="visit_date",
        tolerance=pd.Timedelta(days=merge_tolerance),  # set the tolerance for matching
        direction="nearest",  # find the nearest date, either before or after the key
    )
    # sanity check for duplicated values
    if (
        sum(
            df_out.dropna(subset=["patno", "tiger_id", "visit_date"]).duplicated(
                subset=["patno", "tiger_id", "visit_date"]
            )
        )
        > 0
    ):
        print("!!!! target merge created duplicates, check manually !!!!!")

    return df_out


def historical_growth_velocity(data, params, stage):
    # calculate height velocity as kid
    growth_early_years = (
        data.dropna(subset=params[stage]["height_velocity_column_baby"])
        .groupby("tiger_id")
        .apply(
            find_closest_age,
            targets=[1, 2],
            value_column=params[stage]["height_velocity_column_baby"],
        )
        .reset_index()
    )
    growth_right_before_gh_start = (
        data.dropna(subset=params[stage]["height_velocity_column_treatment"])
        .groupby("tiger_id")
        .apply(
            find_closest_date,
            targets=["gh_dos_date"],
            value_column=params[stage]["height_velocity_column_treatment"],
            n_years=1,
            n_months=0,
        )
        .reset_index()
    )

    birth_height = (
        data[["tiger_id", "birth_length"]]
        .drop_duplicates()
        .rename(columns={"birth_length": 0})
    )
    growth_early_years = pd.merge(growth_early_years, birth_height)
    # calculating height velocity in early years when possible
    growth_early_years["height_velocity_0"] = (
        growth_early_years[1] - growth_early_years[0]
    )
    growth_early_years["height_velocity_1"] = (
        growth_early_years[2] - growth_early_years[1]
    )

    # checking which patients miss data from early on
    missing_early_years = (data.groupby("tiger_id").age.min()).reset_index()
    missing_early_years["missing_1"] = (
        missing_early_years["age"]
        > 1 + params[stage]["height_velocity_missing_tolerance"]
    )

    # making sure that patients who are missing data from early years have velocity NaN
    # TODO: should we also check actual offset from target time?
    growth_early_years = pd.merge(
        growth_early_years,
        missing_early_years,
        how="left",
        on="tiger_id",
        validate="1:1",
    )
    growth_early_years[
        "height_velocity_0"
    ] = growth_early_years.height_velocity_0.where(~growth_early_years.missing_1)
    growth_early_years[
        "height_velocity_1"
    ] = growth_early_years.height_velocity_1.where(~growth_early_years.missing_1)

    # height velocity close to treatment start
    growth_early_years = growth_early_years[
        ["tiger_id", "height_velocity_0", "height_velocity_1"]
    ]
    growth_right_before_gh_start["height_velocity_gh_start_1"] = (
        growth_right_before_gh_start["gh_dos_date-1y_0m"]
        - growth_right_before_gh_start["gh_dos_date"]
    )
    growth_right_before_gh_start = growth_right_before_gh_start.drop(
        columns=["gh_dos_date-1y_0m", "gh_dos_date"]
    )

    early_velocity = pd.merge(
        growth_right_before_gh_start, growth_early_years, on="tiger_id", validate="1:1"
    )

    # height velocity status report
    height_velocity_output = f"""
    # of patients with first data points after 1 yo: {sum(missing_early_years['missing_1'])}
    """

    with open(
        os.path.join(
            params["config"]["paths"]["feature-result"],
            f"height_velocity_missing_data.txt",
        ),
        "w",
    ) as outfile:
        outfile.write(height_velocity_output)
    return early_velocity


def historical_igf1_change(data):
    # find igf_1 values 3m and 12m after treatment start
    igf1_3m_gh_dos_date = (
        data.query("~igf_1.isna()")
        .groupby("tiger_id")
        .apply(
            find_closest_date,
            targets=["gh_dos_date"],
            value_column="igf_1",
            n_months=3,
            direction="forwards",
        )
        .reset_index()
    )
    igf1_1y_gh_dos_date = (
        data.query("~igf_1.isna()")
        .groupby("tiger_id")
        .apply(
            find_closest_date,
            targets=["gh_dos_date"],
            value_column="igf_1",
            n_years=1,
            direction="forwards",
        )
        .reset_index()
    )
    igf1_gh_dos_date = pd.merge(
        igf1_3m_gh_dos_date,
        igf1_1y_gh_dos_date,
        on=["tiger_id", "gh_dos_date"],
        how="outer",
        validate="1:1",
    )
    # clean column names
    igf1_gh_dos_date = igf1_gh_dos_date.rename(
        columns={
            "gh_dos_date": "igf1_gh_dos_date",
            "gh_dos_date+0y_3m": "igf1_gh_dos_date_3m",
            "gh_dos_date+1y_0m": "igf1_gh_dos_date_1y",
        }
    )
    # add column for change in the first 3 months
    igf1_gh_dos_date["change_igf1_gh_dos_date_3m"] = (
        igf1_gh_dos_date["igf1_gh_dos_date_3m"] - igf1_gh_dos_date["igf1_gh_dos_date"]
    )
    # same but as %
    igf1_gh_dos_date["perc_change_igf1_gh_dos_date_3m"] = (
        igf1_gh_dos_date["change_igf1_gh_dos_date_3m"]
        / igf1_gh_dos_date["igf1_gh_dos_date"]
    ) * 100
    # add column for change in the first 1 year
    igf1_gh_dos_date["change_igf1_gh_dos_date_1y"] = (
        igf1_gh_dos_date["igf1_gh_dos_date_1y"] - igf1_gh_dos_date["igf1_gh_dos_date"]
    )
    # same but as %
    igf1_gh_dos_date["perc_change_igf1_gh_dos_date_1y"] = (
        igf1_gh_dos_date["change_igf1_gh_dos_date_1y"]
        / igf1_gh_dos_date["igf1_gh_dos_date"]
    ) * 100
    return igf1_gh_dos_date


def get_dose_proportional_weight(row):
    # get proportional dose to weight
    return row["gh_dose_mg"] / row["weight_kg"]


def get_missing_doses(df, window_size='365d'):
    df['missed_inj'] = df['missed_inj'].fillna(0)
    # total missing doses
    df['missed_inj_cumsum'] = df.groupby('patno')['missed_inj'].cumsum()
    # missing doses per year on GH
    df['missed_inj_per_year_on_gh'] = df['missed_inj_cumsum']/(df['age'] - df['age_gh_start'])
    # missing doses in the last 365 days
    df = df.sort_values(by=['patno', 'visit_date'])
    df[f'missing_inj_last_{window_size}'] = df.groupby('patno', group_keys=False).apply(
        lambda x: x.rolling(window=window_size, on='visit_date', min_periods=1)['missed_inj'].sum())
    return df


def impute_before_first_measurement(group, column, fill_value=0):
    # Find the index of the first non-missing value
    first_measurement_index = group[column].first_valid_index()
    group[f"{column}_imputed"] = group[column]
    # If there's at least one valid measurement
    if first_measurement_index is not None:
        # Replace NaN values before the first non-missing measurement with 0
        group.loc[:first_measurement_index, f"{column}_imputed"] = group.loc[:first_measurement_index, column].fillna(fill_value)
    else:
        group[f"{column}_imputed"] = group[column].fillna(fill_value)
    return group


def find_threshold_for_significant_change(value, span_df, cv_multiplier):
    row = span_df[(span_df['span_start'] <= value) & (span_df['span_end'] > value)]
    if not row.empty:
        return row['cv'].values[0]*cv_multiplier


def flexible_delta(data, column, percent_change_df, cv_multiplier):
    ## delta as proportional change for igf-1
    data = data.sort_values(by=['patno', 'visit_date'])
    data['percentage_change_significant'] = data[column].apply(find_threshold_for_significant_change, span_df=percent_change_df, cv_multiplier=cv_multiplier)

    new_colname = f'{column}_filled'
    data[new_colname] = data.groupby('patno')[column].ffill()
    data['percentage_change'] = data.groupby('patno')[new_colname].transform(lambda x: x.pct_change()*100) # NB: pct_change() actually gives fractional change
    # if increase is higher than X percentage (represented as .X), count as going up
    data[f'{column}_up'] = (data['percentage_change'] > data['percentage_change_significant']).astype(int)
    data[f'{column}_down'] = (data['percentage_change'] < -data['percentage_change_significant']).astype(int)
    data = data.drop(columns=[new_colname, 'percentage_change'])
    return data


def main():
    stage = "features"
    params = yaml.safe_load(open("/workspace/growthcurves/params.yaml"))
    paths = params["config"]["paths"]
    os.chdir(paths["home"])
    os.makedirs(paths[stage], exist_ok=True)
    os.makedirs(paths["feature-result"], exist_ok=True)

    data = pd.read_csv(os.path.join(paths["data"], "pre-processed.csv"))

    puberty_start_corrections = pd.read_csv(
        os.path.join(paths["raw"], "puberty_start_corrections.csv"), sep=";"
    )
    target_data = pd.read_csv(os.path.join(paths["data"], "igf1_processed.csv"))
    target_data = igf1_to_target(target_data, params[stage]["target_column"], params[stage]["target_shift_months"])

    # some clean-up
    data = data.sort_values(["tiger_id", "visit_date"])
    data = convert_dates(data)

    # add in age
    data["age"] = (
        pd.to_datetime(data["visit_date"]) - pd.to_datetime(data["birthday"])
    ).dt.days / 365.25

    ##
    # historical markers
    ##

    # add in height velocity as baby and in the beginning of the treatment
    height_velocity = historical_growth_velocity(data, params, stage)
    data = pd.merge(data, height_velocity, how="left", on="tiger_id")

    # add in changes to IGF-1 in the beginning of the treatment
    igf1_gh_dos_date = historical_igf1_change(data)
    data = pd.merge(data, igf1_gh_dos_date, on="tiger_id", how="left")

    # add puberty start
    puberty_start_dates = find_puberty_start_date(data)
    puberty_start_dates = fix_puberty_start_date(
        puberty_start_dates, puberty_start_corrections
    )
    data = pd.merge(data, puberty_start_dates, on="tiger_id", how="left")

    # calculate parent SDS
    data["father_height_sds"] = (
        data["father_height"] - params[stage]["man_mean"]
    ) / params[stage]["man_sd"]
    data["mother_height_sds"] = (
        data["mother_height"] - params[stage]["woman_mean"]
    ) / params[stage]["woman_sd"]
    data["target_height_sds"] = (
        data["father_height_sds"] + data["mother_height_sds"]
    ) / 2

    # add target height deficit
    data["target_height_deficit_sds"] = data["sd_height"] - data["target_height_sds"]

    # add in treatment phase markers
    data["phase"] = data.apply(determine_treatment_phase, axis=1)

    # add dose proportional to weigh
    data["gh_dose_proportional"] = data.apply(get_dose_proportional_weight, axis=1)

    # add testicle size: it should always be the larger of the testes
    data["testicle_size"] = np.maximum(data["testis_l_ml"], data["testis_r_ml"])
    
    #### imputations
    data = data.sort_values(by=['patno', 'visit_date'])
    # early sex hormones are set to a small constant before first measurement
    data = data.groupby('patno').apply(impute_before_first_measurement, 'testosteron', 0.05).reset_index(drop=True)
    data = data.groupby('patno').apply(impute_before_first_measurement, 'ostradiol', 1).reset_index(drop=True)
    data = data.groupby('patno').apply(impute_before_first_measurement, 'testicle_size').reset_index(drop=True)
    
    # some variables are assumed to stay the same until otherwise indicated
    data['testicle_size_imputed'] = data.groupby('patno')['testicle_size'].ffill()
    data['gh_dose_proportional'] = data.groupby('patno')['gh_dose_proportional'].ffill()
    data['target_height_deficit_sds'] = data.groupby('patno')['target_height_deficit_sds'].ffill()
    data['sd_weight'] = data.groupby('patno')['sd_weight'].ffill()
    
    # add missed injections
    data = get_missing_doses(data)

    ## change units of gh max values to micrograms per liter
    columns_convert_ghmax = data[['gh_max_stimulation', 'gh_max_spontaneous']]
    data[['gh_max_stimulation', 'gh_max_stimulation']] = (columns_convert_ghmax / 2.6).round(1)
    
    # access CV data from params
    cv_multiplier = params[stage]['cv_multiplier'] # how many times CV change do we need to see to assume real change
    
    span_data_testosterone = pd.DataFrame({'span_start':params[stage]['testosterone_change']['span_start'], 
                                           'span_end':params[stage]['testosterone_change']['span_end'], 
                                           'cv':params[stage]['testosterone_change']['cv']})
    
    span_data_ostradiol = pd.DataFrame({'span_start':params[stage]['ostradiol_change']['span_start'], 
                                        'span_end':params[stage]['ostradiol_change']['span_end'], 
                                        'cv':params[stage]['ostradiol_change']['cv']})
    
    span_data_igf1 = pd.DataFrame({'span_start':params[stage]['igf1_change']['span_start'], 
                                    'span_end':params[stage]['igf1_change']['span_end'], 
                                    'cv':params[stage]['igf1_change']['cv']})
    
    span_data_igfbp3 = pd.DataFrame({'span_start':params[stage]['igfbp3_change']['span_start'], 
                                'span_end':params[stage]['igfbp3_change']['span_end'], 
                                'cv':params[stage]['igfbp3_change']['cv']})
    
    span_data_ifg1_igfbp3 = pd.DataFrame({'span_start':params[stage]['igf1_igfbp3_change']['span_start'], 
                                'span_end':params[stage]['igf1_igfbp3_change']['span_end'], 
                                'cv':params[stage]['igf1_igfbp3_change']['cv']})

    data = flexible_delta(data, "igf_1", span_data_igf1, cv_multiplier)
    data = flexible_delta(data, "igfbp_3", span_data_igfbp3, cv_multiplier)
    data = flexible_delta(data, "igf_1_igfbp_3", span_data_ifg1_igfbp3, cv_multiplier) # TODO: this is clearly wrong - how to fix??
    data = flexible_delta(data, "testosteron_imputed", span_data_testosterone, cv_multiplier)
    data = flexible_delta(data, "ostradiol_imputed", span_data_ostradiol, cv_multiplier)
    
    # time delta
    # TODO: do I need to manually remove the original delta column if one-hot encoding?
    delta_dictionary = params[stage]['delta_dict']
    for key in delta_dictionary.keys():
        # TODO: think thorough if this makes sense for dose
        # as dose now is really the future dose
        data = get_values_delta_months(data, key, **delta_dictionary[key])

    ##
    # targets
    ##
    # add in targets
    data = combine_data_with_target(
        data, target_data, params[stage]["target_merge_tolerance"]
    )
    if params[stage]['target_as_delta']:
        today_col = params[stage]["target_column"]
        target_col = "target_" + today_col
        data[target_col] = data[target_col] - data[today_col]

    #
    # save final df
    #
    data.to_csv(os.path.join(paths["features"], "processed.csv"), index=False)

    # helper table for looking at how much complete data we have
    merge_result_table = (
        data.groupby(["phase", "has_igf1", "has_dose_puberty", "has_sex_hormones"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )
    merge_result_table.to_csv(
        os.path.join(paths["feature-result"], "data_completeness_counts_by_phase.csv"),
        index=False,
    )
    # TODO: exclusions based on missing column values, e.g. df.dropna(subset='igf_1')


if __name__ == "__main__":
    main()
