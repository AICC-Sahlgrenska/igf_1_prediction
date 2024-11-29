import yaml
import os
import numpy as np
import pandas as pd
from skimpy import clean_columns

pd.options.mode.chained_assignment = None  # default='warn'


def basic_process(df):
    # drop empty rows
    df = df.dropna(how="all")
    # drop empty columns
    df = df.dropna(how="all", axis=1)
    # clean columnnames
    df = clean_columns(df)
    return df


def clean_up(df):
    # clean datatypes
    if "tiger_id" in df.columns:
        df["tiger_id"] = df.tiger_id.astype(int)
    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(df["visit_date"])
    # drop rows that are complete duplicates
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def convert_comma_to_point(df, column):
    df[column] = df[column].map(
        lambda x: str(x.replace(",", ".")) if isinstance(x, str) else x
    )
    return df


def fix_smaller_than(df, columnname):
    # finds rows with <X in the given column,
    # replaces these with X/2
    # and converts the column to numeric

    # select problematic rows
    contains_less_than = df[columnname].str.contains("<", na=False)
    less_than_df = df.copy().loc[contains_less_than]
    # remove offending character and convert type
    less_than_df["less_than"] = (
        less_than_df[columnname].str.replace("<", "").astype(float)
    )
    less_than_df[columnname] = less_than_df["less_than"] / 2

    # select rows that don't contain <X
    df_not_problematic = df.copy().drop(less_than_df.index, axis=0)
    # combine nonproblematic rows and newly fixed <X rows
    df_out = pd.concat(
        [df_not_problematic, less_than_df.drop(columns=["less_than"])]
    ).sort_index()
    df_out[columnname] = pd.to_numeric(df_out[columnname], errors="coerce")
    return df_out


# Define a function to keep the more complete row for each pair of consecutive days
def keep_most_recent_entries(group):
    # If there's only one row in this group, just return it
    if len(group) == 1:
        return group
    # Otherwise, sort by visit_date descending to have the most recent entries first
    group_sorted = group.sort_values(by="visit_date", ascending=False)
    # Use 'ffill' to forward-fill missing values column-wise with the most recent non-null entry
    group_filled = group_sorted.fillna(method="bfill")
    # Take the first row after filling missing values, which contains the most recent non-null values per column
    return group_filled.iloc[[0]]


def filter_rows(df, df_name, limit_days=1):
    # takes a table and double checks if there are entries for consecutive days
    # if yes, keep the more complete record or, if the records are equally complete, the more recent record
    df = df.sort_values(by=["tiger_id", "visit_date"])

    # Identify where the 'tiger_id' changes
    tiger_id_changes = df["tiger_id"] != df["tiger_id"].shift()

    # Calculate the difference between consecutive dates
    dt_diff = df["visit_date"].diff()

    # Create a boolean series where True indicates either a change in 'tiger_id' or a time gap larger than limit_days
    days_limit = pd.Timedelta(days=limit_days)
    breaks = (dt_diff > days_limit) | tiger_id_changes

    # Create a group identifier that increments every time a break is spotted
    groups = breaks.cumsum()
    # Duplicate grouping for the same 'tiger_id', marking consecutive entries
    duplicates = groups.duplicated(keep=False)
    df_consecutive = df[duplicates]
    
    # apply filtering such that only one row per group is kept
    result_df = (
        df_consecutive.groupby(groups)
        .apply(keep_most_recent_entries)
        .reset_index(drop=True)
    )
    
    df_final = pd.concat([df[~duplicates], result_df])
    df_final = df_final.sort_values(by=["tiger_id", "visit_date"])

    df_consecutive.to_csv(
        f"/workspace/growthcurves/reports/data_processing_reports/problem_rows_{df_name}.csv"
    )

    return df_final


def get_replacement_indices(df, row):
    mask = (df["tiger_id"] == row["tiger_id"]) & (
        df[row["column"]] == row["wrong_value"]
    )
    matching_indices = df.index[mask]
    return matching_indices


def replace_wrong_values(df, corrections):
    # take values from corrections and replace them in df
    for index, row in corrections.iterrows():
        indices = get_replacement_indices(df, row)
        for curr_index in indices:
            df.loc[curr_index, row["column"]] = row["correct_value"]
        # double checking that we did right
        check_indices = get_replacement_indices(df, row)
        if not check_indices.empty:
            raise Exception(
                "something went wrong with replace_wrong_value, are the indices off?"
            )
    df = clean_up(df)
    return df


def remove_problematic_rows(df, df_deletions):
    # making a boolean mask to show if the row is in df_deletions
    mask = df.set_index(["tiger_id", "visit_date"]).index.isin(
        df_deletions.set_index(["tiger_id", "visit_date"]).index
    )
    # only keep rows NOT in deletions
    df_filtered = df[~mask].reset_index(drop=True)
    return df_filtered


def create_merge_success_report(
    left, left_name, right, right_name, df, orphan_right_rows, id_cols, paths
):
    # creates and writes out information about soft merge success
    # does not return anything
    merge_success_info = f"""
    rows in joined table: {len(df)}
    
    LEFT TABLE (larger): {left_name}
    rows in {left_name}: {len(left)}
    rows from {left_name} in the join: {df[left_name + '_visit_date'].dropna().count()}
    Based on visit dates, merge likely created {sum(df.dropna(subset = id_cols + [left_name + '_visit_date']).duplicated(subset=id_cols + [left_name + '_visit_date']))} duplicates in {left_name}

    RIGHT TABLE (smaller): {right_name}
    rows in {right_name}: {len(right)}
    rows from {right_name} in the join: {df[right_name + '_visit_date'].dropna().count()}
    rows in {right_name} missing from join: {len(orphan_right_rows)}
    Based on visit dates, merge likely created {sum(df.dropna(subset=id_cols + [right_name + '_visit_date']).duplicated(subset=id_cols + [right_name + '_visit_date']))} duplicates in {right_name}"""
    with open(
        os.path.join(
            paths["data-result"],
            f"merge_success_left_" + left_name + "_right_" + right_name + ".txt",
        ),
        "w",
    ) as outfile:
        outfile.write(merge_success_info)


def soft_combine(
    df1_in, df1_name, df2_in, df2_name, paths, offset=7, id_cols=["tiger_id", "patno"]
):
    # combines two tables based on id_columns tiger_id and patno
    # and softly on visit_date such that the visit_dates on the two tables
    # can differ by offset number of days
    # lost rows from right table will be automatically saved for inspection

    # make sure that the longer dataframe is always on the left
    if len(df1_in) > len(df2_in):
        left = df1_in
        right = df2_in
        left_name = df1_name
        right_name = df2_name
    else:
        left = df2_in
        right = df1_in
        left_name = df2_name
        right_name = df1_name

    for col in id_cols:
        # merge seems picky about col type in 'by', so we're converting them here
        right[col] = right[col].astype(int)
        left[col] = left[col].astype(int)

    # Ensure 'visit_date' is correct type and that the dfs are sorted by it
    right["visit_date"] = pd.to_datetime(right["visit_date"], format="%Y-%m-%d")
    left["visit_date"] = pd.to_datetime(left["visit_date"], format="%Y-%m-%d")
    right = right.sort_values(by=["visit_date"]).reset_index(drop=True)
    left = left.sort_values(by=["visit_date"]).reset_index(drop=True)

    # for later inspection, save the original visit date per table
    left[left_name + "_visit_date"] = left["visit_date"]
    right[right_name + "_visit_date"] = right["visit_date"]
    left["has_" + left_name] = "yes"
    right["has_" + right_name] = "yes"

    right = right.reset_index().rename(columns={"index": "right_index"})

    # Perform an "asof" merge.
    # This will match each row in df1 with the closest row in df2
    # (where df2's 'visit_date' is less than or equal to df1's 'visit_date')
    df = pd.merge_asof(
        left,
        right,
        by=id_cols,
        on="visit_date",
        tolerance=pd.Timedelta(days=offset),  # set the tolerance for matching
        direction="nearest",  # find the nearest date, either before or after the key
        suffixes=("-" + left_name, "-" + right_name),
    )

    # compare insulin_index in the two tables to find orphan rows
    # make index into set and remove NaNs
    rows_in_right = set({x for x in right.right_index if x == x})
    rows_in_combo = set({x for x in df.right_index if x == x})
    orphan_right_rows = right.iloc[list(rows_in_right - rows_in_combo)].sort_values(
        id_cols
    )
    orphan_right_rows.to_csv(
        os.path.join(paths["data"], "orphan_rows_" + right_name + ".csv"), index=False
    )

    create_merge_success_report(
        left, left_name, right, right_name, df, orphan_right_rows, id_cols, paths
    )
    df = pd.concat([df, orphan_right_rows]).sort_values(by=["tiger_id", "visit_date"])

    df.drop(columns="right_index", inplace=True)
    return df


def clean_merge_result(df):
    # make it clearer which rows have and which don't have which type of info
    for col in df.columns:
        if col.startswith("has"):
            df[col] = df[col].fillna("no")

    # reorganise columns to a more convenient order when we've done all merges
    id_cols = ["tiger_id"]
    visit_cols = [col for col in df.columns if col.endswith("visit_date")]
    presence_cols = [col for col in df.columns if col.startswith("has")]
    start_cols = id_cols + visit_cols + presence_cols
    remaining_cols = [col for col in df.columns if col not in start_cols]
    df = df[start_cols + remaining_cols]
    return df


def process_igf1(df):
    df = basic_process(df)
    df = df.rename(columns={"19": "igf_1_19", "igfbp_3_sds_319": "igfbp_3_sds_19"})
    # Perform a wide to long transformation
    df_long = pd.wide_to_long(
        df,
        stubnames=[
            "visit_date",
            "igf_1",
            "igfbp_3",
            "igf_1_igfbp_3",
            "igf_1_sds",
            "igfbp_3_sds",
            "igf_1_igfbp_3_sds",
        ],
        i=["patno", "tiger_id"],
        j="visit_number",
        sep="_",
    )
    # drop missing values identified by missing date in "visit date"
    df_long = df_long.dropna(subset="visit_date")
    df_long = df_long.reset_index()
    df_long = df_long.drop(columns="visit_number")
    df_long = clean_up(df_long)
    return df_long



def process_igf1_addon(df):
    df = basic_process(df)
    df = df.rename(columns={"datum": "visit_date", "patient_id": "patno"})
    df.drop(columns=["fodelsedatum", "unnamed_8"], inplace=True)
    # for mysterious reasons the patno in this table all start with 1 followed by the real patno,
    # doing a semi-ugly solution to fix this
    df["patno"] = df["patno"] - 10000

    df = clean_up(df)
    return df


def process_insulin(df):
    df = basic_process(df)

    # fix messed up columnnames to enable wide->long transformation
    id_columns = df.columns[0:5]
    # repeating measurements
    measurements = ["visit_date", "hb_a_1c", "f_glucose", "f_insulin"]
    repeats = round((len(df.columns) - len(id_columns)) / len(measurements))
    # get new repeated measurement names
    measurements_with_suffixes = [
        f"{string}_{i+1}" for i in range(repeats) for string in measurements
    ]
    new_colnames = list(id_columns) + list(measurements_with_suffixes)
    df.columns = new_colnames

    # convert wide to long
    df_long = pd.wide_to_long(
        df,
        stubnames=["visit_date", "hb_a_1c", "f_glucose", "f_insulin"],
        i=id_columns,
        j="visit_number",
        sep="_",
    ).reset_index()

    # drop missing values identified by missing date in "visit date"
    df_long = df_long.dropna(subset="visit_date")
    df_long = df_long.reset_index(drop=True)
    df_long = df_long.replace(",", ".", regex=True)

    # fix funky values & convert to numeric
    df_long = fix_smaller_than(df_long, "f_insulin")
    # convert the other columns to numeric too
    df_long["hb_a_1c"] = pd.to_numeric(df_long["hb_a_1c"], errors="coerce")
    df_long["f_glucose"] = pd.to_numeric(df_long["f_glucose"], errors="coerce")
    df_long = df_long.drop(columns=["birthday", "sex", "gh_start_date", "visit_number"])

    df_long = clean_up(df_long)
    return df_long


def process_hormones(df):
    df = basic_process(df)
    df = convert_comma_to_point(df, "testosteron")

    df = df.rename(
        columns={
            "datum_ostradiol": "visit_date",
            "tidpunkt_1": "visit_time",
        }
    )
    df = df.drop(columns=["fodelsedata", "datum_testosteron", "tidpunkt", "patnr"])

    df = fix_smaller_than(df, "ostradiol")
    df = fix_smaller_than(df, "testosteron")
    df = clean_up(df)
    return df


def process_dose_puberty(df):
    df = basic_process(df)
    df = df.drop(columns=["puberty", "gh_year"])
    # dose puberty has a lot of rows with no data, let's drop anything that only has identifiers and date
    # ie demand at least 3 non-nan values
    df = df.dropna(thresh=3)
    df = clean_up(df)
    return df


def process_gh_max(df):
    df = basic_process(df)
    df = df.rename(columns={"gh_max_stimulation_m_u_l": "gh_max_stimulation", 
                            "gh_max_spontaneous_m_u_l": "gh_max_spontaneous"})
    df.drop(columns=["maintenance_date"], inplace=True)
    df = clean_up(df)
    return df


def process_GH_End(df):
    df = df.rename(columns={"Pat no ": "patno", "Weight": "final_weight", "Height": "final_height", "DateMeasured": "final_date_measured"})
    df = df.drop(columns={"Unnamed: 2"})
    df = basic_process(df)
    df["final_date_measured"] = df["final_date_measured"].str[0:10]
    df = clean_up(df)
    return df


def process_basic_data(df):
    df = basic_process(df)

    is_male = df['sex']
    is_male = pd.get_dummies(is_male)
    df["is_male"] = is_male['Male']

    df = clean_up(df)
    
    return df


def process_sds_data(df):
    df = basic_process(df) 
    df = df.drop(columns=["sex","sex_1", "sex_2", "sex_3", "age", "age_1", "bmi", "bmi_1", "weight_kg", "weight_kg_1", "height_cm", "height_cm_1", "birthday"])    
    df = df.rename(columns={"id": "tiger_id", "age_2": "age", "sd": "sd_weight", "sd_1": "sd_height", "sd_2": "sd_bmi"})
    df = clean_up(df)
    return df

def add_age_GH_start(df):
    gh_start = pd.to_datetime(df['gh_start_date'])
    birthday = pd.to_datetime(df['birthday'])
    
    days = (gh_start - birthday).dt.days
    age = np.round(days/365.24, 2)

    df.insert(len(df.columns), "age_gh_start", age)
    return df


def process_early_growth_addon(df):
    df = basic_process(df)
    df = df.rename(columns={"datum":"visit_date",
                            "langd":"height_cm",
                            "vikt_kg":"weight_kg"})
    df['visit_date'] = pd.to_datetime(df['visit_date'], yearfirst= True, format='%y%m%d')
    df = df.drop(columns={'age', "pat_no"})

    df = clean_up(df)
    return df


def main():
    stage = "data"
    params = yaml.safe_load(open("/workspace/growthcurves/params.yaml"))
    paths = params["config"]["paths"]
    os.chdir(paths["home"])
    os.makedirs(paths[stage], exist_ok=True)
    os.makedirs(paths["data-result"], exist_ok=True)

    # import data
    data_igf1 = pd.read_csv(os.path.join(paths["raw"],'IGF-1_IGFBP3.csv'), delimiter = ';', decimal=',')
    data_igf1_addon = pd.read_csv(os.path.join(paths["raw"],'kompletterande IGF-1.csv'), delimiter = ';', decimal=',')
    data_igf1_corrections = pd.read_csv(os.path.join(paths["raw"],'igf-1_corrections.csv'), delimiter = ';', decimal=',')
    data_insulin = pd.read_csv(os.path.join(paths["raw"],'HbA1c_glucose_insulin.csv'), delimiter = ';', decimal=',')
    insulin_corrections = pd.read_csv(os.path.join(paths["raw"],'HbA1c_glucose_insulin_corrections.csv'))
    data_hormones = pd.read_csv(os.path.join(paths["raw"],'sex_hormones_cleaned.csv'), delimiter = ';', decimal=',')
    data_dose_puberty = pd.read_csv(os.path.join(paths["raw"],'height_weight_puberty_GH_dose.csv'), delimiter = ';', decimal=',')
    dose_puberty_corrections = pd.read_csv(os.path.join(paths["raw"],'height_weight_puberty_GH_dose_corrections.csv'))
    dose_early_growth_addon = pd.read_csv(os.path.join(paths["raw"],'saknar_tidig_tillvaxt.csv'), 
                                          delimiter = ';', decimal=',',
                                          dtype={'datum': 'str'})

    data_GH_End = pd.read_csv(os.path.join(paths["raw"],'Correct_slutläng&GHStop.csv'), delimiter = ';', decimal=',')
    df_final_height_corrections = pd.read_csv(os.path.join(paths["raw"],'slutlängd_corrections.csv'))

    dose_puberty_deletions = pd.read_csv(
        os.path.join(paths["raw"], "puberty_dose_delete_rows.csv"),
        delimiter=";",
        decimal=",",
    )

    data_gh_max = pd.read_csv(
        os.path.join(paths["raw"], "maintenance_date_and_GH_max.csv"), delimiter=";", decimal=",")
    

    data_basic = pd.read_csv(
        os.path.join(paths["raw"], "basic_data.csv"), 
        delimiter=";", 
        decimal=","
    )
    data_sds_weight_height = pd.read_csv(
        os.path.join(paths["raw"], "weight_height_SDS.csv"), 
        delimiter=";", 
        decimal=","
    )

    # pre-process data
    data_igf1 = process_igf1(data_igf1)
    data_igf1_addon = process_igf1_addon(data_igf1_addon)
    insulin = process_insulin(data_insulin)
    data_hormones = process_hormones(data_hormones)
    dose_puberty = process_dose_puberty(data_dose_puberty)
    data_GH_End = process_GH_End(data_GH_End)
    data_basic = process_basic_data(data_basic)
    data_sds_weight_height = process_sds_data(data_sds_weight_height)
    data_gh_max = process_gh_max(data_gh_max)
    dose_early_growth_addon = process_early_growth_addon(dose_early_growth_addon)

    # add age at GH-start
    data_basic = add_age_GH_start(data_basic)

    # join additional new data tables 
    igf1 = pd.concat([data_igf1, data_igf1_addon]).sort_values(
        by=["tiger_id", "visit_date"]
    ).drop(columns='patno')
    df_background = data_basic.merge(data_GH_End, on="patno", validate="1:1")
    df_background = df_background.merge(data_gh_max, on=["patno","tiger_id"], validate="1:1")
    dose_puberty = dose_puberty.merge(dose_early_growth_addon, on=['tiger_id','visit_date', "height_cm", "weight_kg"], how='outer')
    
    # go through known errors and replace
    insulin = replace_wrong_values(insulin, insulin_corrections)
    dose_puberty = replace_wrong_values(dose_puberty, dose_puberty_corrections)
    dose_puberty = remove_problematic_rows(dose_puberty, dose_puberty_deletions)
    igf1 = replace_wrong_values(igf1, data_igf1_corrections)
    df_background = replace_wrong_values(df_background, df_final_height_corrections)

    # remove near-duplicates (entries on consecutive days or very close in time)
    # only one entry within filter_tolerance days is kept
    igf1 = filter_rows(igf1, "igf1", params[stage]["merge_tolerance_hormones"])
    insulin = filter_rows(insulin, "insulin", params[stage]["merge_tolerance_hormones"])
    data_hormones = filter_rows(data_hormones, "data_hormones", params[stage]["merge_tolerance_hormones"])
    dose_puberty = filter_rows(dose_puberty, "dose_puberty", params[stage]["merge_tolerance_length"])
    data_sds_weight_height = filter_rows(data_sds_weight_height, "sds", params[stage]["merge_tolerance_length"])

    # save interim files
    igf1.to_csv(os.path.join(paths[stage], "igf1_processed.csv"), index=False)
    insulin.to_csv(os.path.join(paths[stage], "insulin_processed.csv"), index=False)
    data_hormones.to_csv(
        os.path.join(paths[stage], "sex_hormones_processed.csv"), index=False
    )
    dose_puberty.to_csv(
        os.path.join(paths[stage], "dose_puberty_processed.csv"), index=False
    )
    df_background.to_csv(
        os.path.join(paths[stage], "background_processed.csv"), index=False
    )
    data_sds_weight_height.to_csv(
        os.path.join(paths[stage], "data_sds_weight_height.csv"), index=False
    )
    # # make more and more joins
    # we start by merging all of the hormones, as these have a narrower merge tolerance
    all_hormones = soft_combine(
        igf1,
        "igf1",
        data_hormones,
        "sex_hormones",
        paths,
        offset=params[stage]["merge_tolerance_hormones"],
        id_cols=["tiger_id"],
    )
    all_hormones.to_csv(
        os.path.join(paths[stage], "combined_all_hormones.csv"), index=False
    )

    # combine dose data and sds data
    processed_data_dose_sds = soft_combine(
        dose_puberty,
        "dose_puberty",
        data_sds_weight_height,
        "all_sds_data",
        paths,
        offset=params[stage]["merge_tolerance_length"],
        id_cols=["tiger_id"],
    )
    
    processed_data_dose_sds.to_csv(
        os.path.join(paths[stage], "combined_dose_sds.csv"), index=False
    )
    
    # then add in all_hormones, with a larger merge tolerance
    processed_data = soft_combine(
        processed_data_dose_sds,
        "dose_and_sds",
        all_hormones,
        "all_hormones",
        paths,
        offset=params[stage]["merge_tolerance_length"],
        id_cols=["tiger_id"],
    )

    processed_data.to_csv(
        os.path.join(paths[stage], "combined_dose_puberty_hormones.csv"), index=False
    )

    # problematic duplicate rows in the processed data
    duplicates = processed_data.dropna(subset="all_hormones_visit_date").loc[
        processed_data.dropna(subset="all_hormones_visit_date").duplicated(
            subset=["tiger_id", "all_hormones_visit_date"], keep=False
        )
    ]
    duplicates = duplicates.sort_values(["tiger_id", "visit_date"])
    # save problematic rows
    duplicates.to_csv(os.path.join(paths[stage], "merged_data_duplicate_rows.csv"))
    # duplicate rows seem to disappear with proper early filtering, 
    # keeping a warning in case that does not happen
    if(len(duplicates)>0):
        print('YOU HAVE DUPLICATES IN THE DATA AGAIN')
    
    processed_data = clean_merge_result(processed_data)

    # finally, combine long data with background
    data_out = pd.merge(df_background, processed_data, on=["tiger_id"])

    # final version of csv to save
    data_out.to_csv(os.path.join(paths[stage], "pre-processed.csv"), index=False)

    # helper table for looking at how much complete data we have
    merge_result_table = (
        data_out.groupby(
            ["has_igf1", "has_dose_puberty", "has_sex_hormones"]
        )
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )
    merge_result_table.to_csv(
        os.path.join(paths["data-result"], "merge_results_counts.csv"), index=False
    )


if __name__ == "__main__":
    main()
