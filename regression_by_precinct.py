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

# precincts = [106, 22, 1, 6, 68, 94, 48, 18, 10, 17, 108, 78, 26, 72, 61, 100, 115, 63, 110, 24, 81, 28, 5, 20, 45, 33, 50, 30, 123, 103, 7, 88, 46, 47, 77, 19, 34, 32, 43, 76, 49, 13, 104, 111, 52, 71, 83, 14, 9, 70, 90, 62, 112, 113, 60, 42, 23, 25, 69, 73, 109, 40, 75, 122, 66, 41, 114, 84, 102, 67, 44, 107, 79, 101, 105, 121, 120]
precincts = [106]

# Create dummy columns
dummy_race = pd.get_dummies(df['race'], prefix='race')
dummy_sex = pd.get_dummies(df['sex'], prefix='sex')


# create a clean data frame for the regression
# cols_to_keep = ["pct", "frisked", "othpers", "cs_objcs", "cs_descr",
#               "cs_casng", "cs_lkout", "cs_cloth", "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other"]
cols_to_keep = ["pct", "frisked", "othpers", "cs_objcs", "cs_bulge"]

# Join dummy columns with original clean dataset
data = df[cols_to_keep].join(dummy_race.ix[:, :'race_W'])
data = data.join(dummy_sex['sex_0'])


# Add the intercept
data['intercept'] = 1.0

# ------------------   METHOD 1 - Full Analysis  ------------------

for pct in precincts:
    # pct_data = data
    pct_data = data[data['pct']==pct]
    pct_data = pct_data.ix[:, 'frisked':]
    corr = np.corrcoef(pct_data, rowvar=0)
    print corr

    print pct_data.describe

    # define training columns
    train_cols = pct_data.columns[1:]
    print train_cols

    print np.linalg.matrix_rank(data[train_cols].values)

    # define model with variable to predict
    logit = sm.Logit(pct_data['frisked'], pct_data[train_cols])

    result = logit.fit(method='bfgs')

    # fit the model
    # try:
    #     result = logit.fit()
    # except:
    #     break

    output_file = 'pct_frisked/pct_' + pct.__str__() + '.csv'

    output = open(output_file, mode='w')
    output.write(result.summary().as_csv())
    output.close()

# # ------------------   METHOD 2 - Regression with only low p-value features  ------------------
#
# # Define low p-value features
# cols_to_keep = ['intercept', 'cs_drgtr', 'cs_lkout', 'othpers', 'cs_descr', 'cs_casng', 'cs_cloth', 'cs_furtv', 'cs_vcrim', 'cs_bulge', 'cs_other', 'race_B', 'sex_0', 'pct_7', 'pct_9', 'pct_10', 'pct_13', 'pct_19', 'pct_23', 'pct_25', 'pct_26', 'pct_28', 'pct_30', 'pct_32', 'pct_33', 'pct_34', 'pct_41', 'pct_42', 'pct_43', 'pct_44', 'pct_45', 'pct_46', 'pct_47', 'pct_48', 'pct_50', 'pct_52', 'pct_60', 'pct_61', 'pct_63', 'pct_66', 'pct_67', 'pct_69', 'pct_70', 'pct_71', 'pct_72', 'pct_75', 'pct_79', 'pct_81', 'pct_83', 'pct_90', 'pct_100', 'pct_101', 'pct_103', 'pct_104', 'pct_105', 'pct_106', 'pct_107', 'pct_108', 'pct_109', 'pct_110', 'pct_111', 'pct_112', 'pct_113', 'pct_114', 'pct_115', 'pct_120', 'pct_121', 'pct_122']
#
# # define model with variable to predict
# logit = sm.Logit(data['frisked'], data[cols_to_keep])
#
# # fit the model
# result = logit.fit()
# print result.summary()
#
# # # ALTERNATE REGRESSION MODEL
# # clf = linear_model.LogisticRegression()
# # clf.fit(data[train_cols], data['frisked'])
# # print clf.coef_