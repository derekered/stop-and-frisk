__author__ = 'derek'

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# # Library for alternate regression technique
# from sklearn import linear_model


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

# print df.head
# print df.columns
# print df.dtypes
# print df.describe()

# Create dummy columns
dummy_race = pd.get_dummies(df['race'], prefix='race')
dummy_sex = pd.get_dummies(df['sex'], prefix='sex')
dummy_pct = pd.get_dummies(df['pct'], prefix='pct')

# create a clean data frame for the regression
cols_to_keep = ["frisked", "othpers", "cs_objcs", "cs_descr",
              "cs_casng", "cs_lkout", "cs_cloth", "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other"]


# Join dummy columns with original clean dataset
data = df[cols_to_keep].join(dummy_race['race_B'])
data = data.join(dummy_sex['sex_0'])
data = data.join(dummy_pct.ix[:, :'pct_122'])


# Add the intercept
data['intercept'] = 1.0

# ------------------   METHOD 1 - Full Analysis  ------------------
# define training columns
train_cols = data.columns[1:]

# define model with variable to predict
logit = sm.Logit(data['frisked'], data[train_cols])


# fit the model
result = logit.fit()

print result.summary()

# ------------------   METHOD 2 - Regression with only low p-value features  ------------------

# Define low p-value features
cols_to_keep = ['intercept', 'cs_drgtr', 'cs_lkout', 'othpers', 'cs_descr', 'cs_casng', 'cs_cloth', 'cs_furtv', 'cs_vcrim', 'cs_bulge', 'cs_other', 'race_B', 'sex_0', 'pct_7', 'pct_9', 'pct_10', 'pct_13', 'pct_19', 'pct_23', 'pct_25', 'pct_26', 'pct_28', 'pct_30', 'pct_32', 'pct_33', 'pct_34', 'pct_41', 'pct_42', 'pct_43', 'pct_44', 'pct_45', 'pct_46', 'pct_47', 'pct_48', 'pct_50', 'pct_52', 'pct_60', 'pct_61', 'pct_63', 'pct_66', 'pct_67', 'pct_69', 'pct_70', 'pct_71', 'pct_72', 'pct_75', 'pct_79', 'pct_81', 'pct_83', 'pct_90', 'pct_100', 'pct_101', 'pct_103', 'pct_104', 'pct_105', 'pct_106', 'pct_107', 'pct_108', 'pct_109', 'pct_110', 'pct_111', 'pct_112', 'pct_113', 'pct_114', 'pct_115', 'pct_120', 'pct_121', 'pct_122']

# define model with variable to predict
logit = sm.Logit(data['frisked'], data[cols_to_keep])

# fit the model
result = logit.fit()

print result.summary()

# # ALTERNATE REGRESSION MODEL
# clf = linear_model.LogisticRegression()
# clf.fit(data[train_cols], data['frisked'])
# print clf.coef_