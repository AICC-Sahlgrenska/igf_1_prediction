IGF-1 prediction in children with growth hormone deficiency
==============================

The codebase associated with paper [manuscript ID to come after acceptance], available [link to come after acceptance]. 

## Working in the codebase


The code is conceptualised as a [dvc](https://dvc.org/) pipeline, which is a simple Directed Acyclic Graph (DAG) pipeline defining the data handling steps from data import all the way to model outputs. All of the steps happen automatically when the pipeline gets triggered, which allows you as a reader to see exactly how the analysis was conducted.

These steps are as follows (these are defined in the [dvc.yaml file](https://github.com/AICC-Sahlgrenska/igf_1_prediction/blob/main/dvc.yaml)):
1. data: Data files are loaded from a separate data share, simple data cleaning (e.g. adjusting variable names) is done and the different data files are combined.
2. features: When a feature needs to be calculated (i.e. it was not present in the original data file in the correct format), it gets calculated here. For example, all features that show change (for example, change in target height deficit) need to be calculated here.
3. prepare_modelling: Select features used for modelling and create Table1. Exclude data with missing values and split data into testing and training set.
4. train: Training Linear Regression model and Explainable Boosting Regressor.
5. train_symbolic: Training Symbolic Regression.
6. evaluate: Evaluate the models from steps 4 and 5 against the test set and collate the result.

For the manuscript, outputs are also taken from:

- notebooks/ebm_explanations.ipynb: a jupyter notebook for making and cleaning up Explainable Boosting Machine plots.
- notebooks/make_manuscript_outputs.ipynb: a jupyter notebook for inspecting the properties of Explainable Boosting Machine feature explanations.
- notebooks/predictions_visualisation.ipynb: a jupyter notebook for visually inspecting prediction outputs vs ground truth in the test data.

## Setting up the development environment

This project combines two analytical environments:
1. the conda environment called 'gpg' is the one needed to build [the gpg package](https://github.com/marcovirgolin/gpg.git), which in the article is used to do Symbolic Regression.
2. the conda environment called 'gpg-modified' is then used to run all of the actual analyses. 

Getting the environments set up right is a bit tricky but following these steps to the t should help. If not, please contact the authors who will be happy to help.

This project has been developed using Region Västra Götaland's AI Platform, which uses a fully dockerized system. Some of the environment set-up has been optimized for that, but you should be able to get to the same end result using 

### Getting the development environment to work
In a new job, run 

1. Install miniconda:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

```
2. Clone and make gpg
``` 
git clone https://github.com/marcovirgolin/gpg.git
cd gpg
conda env create -f environment.yml
conda init
conda activate gpg
make 
````
3. To then develop with the rest of the codebase, you need to run the following:
```bash
cd /workspace
git clone https://github.com/AICC-Sahlgrenska/igf_1_prediction.git
cd growthcurves
conda env create -f environment.yml 
conda activate gpg-modified
pip uninstall nvidia_cublas_cu11
```

3.  In your job, set python interpreter to wherever your gpg-modified environment got saved (for us it was /workspace/.conda/envs/gpg-modified/bin/python)

### Errors in Python jobs
Sometime the conda installation will throw the following error: "ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /workspace/miniconda3/envs/gpg-modified/lib/python3.9/site-packages/pandas/_libs/window/aggregations.cpython-39-x86_64-linux-gnu.so)". To fix it, do the following:

1. Add this line to your .bashrc `export LD_LIBRARY_PATH=/workspace/miniconda3/lib`

2. run `source ~/.bashrc` or open a new terminal window

3. re-activate conda environment

## Decisions

We want to collate all decisions that deal with excluding data and decisions that can have a big impact on the final model. This was originally a working document for internal use but we've chosen to leave it up for open review to allow for more nuanced understanding of the limitations and analytical choices in the paper.

### Data and analytical choices

Decision log on the data and analytical choices embedded in the code

| Date | Decision | Reasoning |
| ------ | ------ |------ | 
| 2024-07-18 | We will model igf1 in SDS, not natural units. | This is not what we said originally, but it will be easier to discuss performance improvements in SDS ("our model improves the known normalization") vs having to defend against reviewers saying we already knew that age matters, how is your approach novel, even if the mechanistic interpretability suffers. | 
| 2024-07-18 | We will predict t0+3m and t0+12m. | These time windows are the ones where we have reasonable number of data points (t0+6m is much less). Initially, we'll model both and compare and contrast the models in the discussion (XYZ is driving the prediction at 3m but not at 12m, why is that) | 
| 2024-06-01 | We will primarily model boys. | Too many of the used features (sex hormones, childhood growth, puberty markers) are sex dependent and we don't have anywhere nearly enough data for girls, unfortunately. | 
| 2024-02-06 | We will initially treat maintenance as a single phase, and use puberty markers as features instead of trying to divide it into maintenance-growth and puberty-growth | Depending on results, we may need to still separate pre-puberty and puberty, which is a bridge we'll cross if we get there | 
| 2024-02-06 | The model we will try to fit takes the features as defined below at t0 and tries to predict IGF-1 level proportion at t0+6 months. | This time frame was chosen to be concordant with existing clinical praxis of follow-up every 6 months (even if we have more frequent data available) (note that this decision was later overruled) | 
| 2023-12-21 | We will primarily be building a prediction model | After a long discussion about statistical and machine learning mindset, the doctos decided that it will be most interesting to conduct project from a ML mindset | 
| 2023-10-01 | The primary outcome measure will be unadjusted IGF-1 | IGF-1 SDS already takes into account a number of our variables, and it is not clear how good the normalization is (note that this decision was later overruled) | 

### Features
# TODO Full list of features with their definitions
This is a living table about features and how they are defined, to be updated every time we merge a feature branch. 

All features for modeling of maintenance phase as discussed 2024-02-06: 
- age (decimals)
- gh max (<- data currently missing!!)
- Tanner stage at t0
- delta (Tanner stage at t0 - 6 months)
- sex hormones at t0
- delta (sex hormones at t0 - 12 months)
- heightSDS/(average parents' heightSDS) at t0
- delta (heightSDS/(average parents' heightSDS) at t0 - 6 months)
- height velocity (SDS) at 1yo, 2yo and the last year prior to GHStart 
- weight velocity (SDS) at 1yo, 2yo 
- weightSDS at t0
- delta (weightSDS at t0 - 6 months)
- whether the patient has historically missed many doses (this needs further refinement)

| Feature name | Definition | 
| ------ | ------ |------ | 
| **Early life features** |
| birth weight | in grams | 
| birth height | in cm |
| height_velocity_0 | Growth first year of life (cm) | 
| height_velocity_1 | Growth second year of life (cm) | 
| **Early treatment features** |
| age treatment start | Age at treatment start |
| gh_max | Stimulated GH peak (μg/L) |
|  height_velocity_gh_start-1 | change in height over 1 year immediately prior to GH treatment onset, defined as change in SDS |
| delta IGF-1 early treatment 3 months | change in IGF-1 at first 3 months of treatment | 
| delta IGF-1 early treatment 12 months | change in IGF-1 at first 12 months of treatment |
| **Features related to prediction time (t0)** |
| age | years (incl decimals) |
| weight | weight at t0, in SDS | 
| delta_weight | Change in weight past 12 months (SDS) |
| target_height_deficit_sds | Target height deficit (SDS) |
| delta_target_height_deficit_sds | Change in target height deficit past 12 months (SDS) |
| gh_dos t0 | GH dose (mg/kg/day) | 
| delta gh_dos | Change in GH dose past 3 months (mg/kg/day) |
| IGF-1 | IGF-1 value at t0 |
| delta IGF-1 | Was there a significant change IGF-1 level in the 12 months prior to t0? (see paper for a more thorough discussion on how we defined a significant change) |
| delta IGFBP3 |  Was there a significant change IGFBP3 level in the 12 months prior to t0? (see paper for a more thorough discussion on how we defined a significant change) |
| IGF-1/IGFBP3 | IGF-1/IGFBP3 proportion at t0 |
| delta IGF-1/IGFBP3 | Was there a significant change  IGF-1/IGFBP3 level in the 12 months prior to t0? (see paper for a more thorough discussion on how we defined a significant change) |
| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |
| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |
| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |


| phase | Phase describes the phase of the treatment. The currently implemented phases are pre-treatment: beginning of data – GHDosDate ; catch-up: GHDosDate – GHDosMDate ; maintenance-growth: GHDosMDate – Puberty start date; pubertal-growth: Puberty start date – GH stop date; post-treatment: GH stop date - end of data | We will at least initially treat maintenance-growth and pubertal-growth as a single phase; this may need to be revised later |
| puberty_start_date | puberty start is defined as the first date when a boy has testis size 4.0ml or larger or the first date when a girl had >= 2 in the Breast&genitalia tanner stadium rating | some subjects went up and down in testis size in a curious pattern, a corrections file was used for these |



| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |
| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |
| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |
| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |
| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |
| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |
| gh_dos | GH dose is defined as µg/kg/dygn | We expect this to be the same for everyone in catch-up phase; some changes for some participants when we reach maintenance phase |


### Imputation methods
| Feature name | Imputation strategy | Comments? | 
| ------ | ------ |------ | 
| Testosterone | 0.05 nmol/L | Testosterone was first measured from 2 years before puberty was reached. WWe impute values from before then with limit of detection (LOD)/2 | 
| Ostradiol | 0.05 nmol/L | Ostradiol was first measured from 2 years before puberty was reached. We impute values from before then with limit of detection (LOD)/2 |
| testicle size | forward fill | The assumption is that the testicle size does not generally shrink during puberty, so values between visits where testicle size was estimated were imputed to the value of the previous visit |
| GH dose | forward fill | The assumption is that on the visits when no new GH dose is set, the dosing continues as the same |
| target height deficit | forward fill | Not all visits include measuring of height. We assume that target height deficit stays the same between two visits, and only gets updated when the child is measured again. |
| weight (SDS) | forward fill | Not all visits include measuring of weight. We assume that the weight SDS stays the same between two visits, and only gets updated when the child is measured again. |



## Project Organization
<details><summary>Click to expand</summary>

------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── notebooks          <- Notebooks used to generate some of the figures in the final manuscript
    ├── reports            <- Generated outputs, such as tables and figures end up here
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── growthcurves   <- Source code for use in this project.
    │   ├── __init__.py    <- Makes growthcurves a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── prepare_modeling <- Scripts to prep data for modeling
    │   │                     
    │   └── modeling         <- Scripts to train models and then use trained models to make
    │                         predictions and then evaluate these predictions
    |
    ├── Dockerfile         <- Dockerfile with settings to run scripts in Docker container
    ├── dvc.yaml           <- DVC pipeline; see dvc.org
    └── params.yaml        <- Parameter values used by DVC pipeline

--------
</details>



<p><small>Project based on the <a target="_blank" href="https://github.com/Vastra-Gotalandsregionen/data-science-template">data science project template</a>. #cookiecutterdatascience.</small></p>
