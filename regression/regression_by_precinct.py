__author__ = 'derek'

import pandas as pd
import statsmodels.api as sm
import numpy as np


# Read the data
df = pd.read_csv("CleanData.csv", low_memory=False)
df.columns = ["pct", "datestop", "timestop", "inout", "trhsloc", "typeofid", "explnstp", "othpers", "arstmade",
              "offunif", "officrid", "frisked", "searched", "contrabn", "pistol", "riflshot", "asltweap", "knifcuti",
              "machgun", "othrweap", "forceused", "rf_vcrim", "rf_othsw", "ac_proxm", "rf_attir", "cs_objcs", "cs_descr",
              "cs_casng", "cs_lkout", "rf_vcact", "cs_cloth", "cs_drgtr", "ac_evasv", "ac_assoc", "cs_furtv",
              "rf_rfcmp", "ac_cgdir", "rf_verbl", "cs_vcrim", "cs_bulge", "cs_other", "ac_incid", "ac_time", "rf_knowl",
              "ac_stsnd", "ac_other", "sb_hdobj", "sb_outln", "sb_admis", "sb_other", "rf_furt", "rf_bulg", "offverb",
              "offshld", "sex", "race", "age", "ht_feet", "ht_inch", "weight", "haircolr", "eyecolor", "build",
              "othfeatr", "detailCM"]

precincts = [106, 22, 1, 6, 68, 94, 48, 18, 10, 17, 108, 78, 26, 72, 61, 100, 115, 63, 110, 24, 81, 28, 5, 20, 45, 33, 50, 30, 123, 103, 7, 88, 46, 47, 77, 19, 34, 32, 43, 76, 49, 13, 104, 111, 52, 71, 83, 14, 9, 70, 90, 62, 112, 113, 60, 42, 23, 25, 69, 73, 109, 40, 75, 122, 66, 41, 114, 84, 102, 67, 44, 107, 79, 101, 105, 121, 120]
# precincts = [106]

# Create dummy columns
dummy_race = pd.get_dummies(df['race'], prefix='race')
dummy_sex = pd.get_dummies(df['sex'], prefix='sex')


# ------------------   FRISK ANALYSIS  ------------------

# create a clean data frame for the regression
cols_to_keep = ["pct", "frisked", "othpers", "cs_objcs", "cs_descr",
              "cs_casng", "cs_lkout", "cs_cloth", "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other"]

# Join dummy columns with original clean dataset
data = df[cols_to_keep].join(dummy_race.ix[:, :'race_W'])
data = data.join(dummy_sex['sex_0'])


# Add the intercept
data['intercept'] = 1.0

# Run regression for each precinct
for pct in precincts:
    pct_data = data[data['pct']==pct]
    pct_data = pct_data.ix[:, 'frisked':]

    # define training columns
    train_cols = pct_data.columns[1:]

    # define model with variable to predict
    logit = sm.Logit(pct_data['frisked'], pct_data[train_cols])

    # fit the model
    try:
        result = logit.fit(maxiter=1000)
    except:
        continue

    # Write results to output csv
    output_file = 'pct_frisked/pct_' + pct.__str__() + '.csv'

    output = open(output_file, mode='w')
    output.write(result.summary().as_csv())
    output.close()

# ------------------   SEARCH ANALYSIS  ------------------

# create a clean data frame for the regression
cols_to_keep = ["pct", "searched", "othpers", "cs_objcs", "cs_descr",
              "cs_casng", "cs_lkout", "cs_cloth", "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other"]

# Join dummy columns with original clean dataset
data = df[cols_to_keep].join(dummy_race.ix[:, :'race_W'])
data = data.join(dummy_sex['sex_0'])


# Add the intercept
data['intercept'] = 1.0


# Run regression for each precinct
for pct in precincts:
    pct_data = data[data['pct']==pct]
    pct_data = pct_data.ix[:, 'searched':]

    # define training columns
    train_cols = pct_data.columns[1:]

    # define model with variable to predict
    logit = sm.Logit(pct_data['searched'], pct_data[train_cols])

    # fit the model
    try:
        result = logit.fit(maxiter=1000)
    except:
        continue

    output_file = 'pct_searched/pct_' + pct.__str__() + '.csv'

    output = open(output_file, mode='w')
    output.write(result.summary().as_csv())
    output.close()