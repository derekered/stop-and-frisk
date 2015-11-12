__author__ = 'derek'

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# read the data in
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

# dummy_id = pd.get_dummies(df['typeofid'], prefix='typeofid')
# dummy_inout = pd.get_dummies(df['inout'], prefix='inout')
# dummy_trhsloc = pd.get_dummies(df['trhsloc'], prefix='trhsloc')
dummy_race = pd.get_dummies(df['race'], prefix='race')
dummy_sex = pd.get_dummies(df['sex'], prefix='sex')
dummy_pct = pd.get_dummies(df['pct'], prefix='pct')
# dummy_ = pd.get_dummies(df[''], prefix='')
# dummy_ = pd.get_dummies(df[''], prefix='')
# dummy_ = pd.get_dummies(df[''], prefix='')


# print dummy_race.head()
# print dummy_pct.head()


# create a clean data frame for the regression
cols_to_keep = ["frisked", "searched", "contrabn", "othpers", "forceused"]

# cols_to_keep = ["explnstp", "othpers", "arstmade",
#               "offunif", "officrid", "frisked", "searched", "contrabn", "pistol", "riflshot", "asltweap", "knifcuti",
#               "machgun", "othrweap", "pf_hands", "pf_wall", "pf_grnd", "pf_drwep", "pf_ptwep", "pf_baton", "pf_hcuff",
#               "pf_pepsp", "pf_other", "rf_vcrim", "rf_othsw", "ac_proxm", "rf_attir", "cs_objcs", "cs_descr",
#               "cs_casng", "cs_lkout", "rf_vcact", "cs_cloth", "cs_drgtr", "ac_evasv", "ac_assoc", "cs_furtv",
#               "rf_rfcmp", "ac_cgdir", "rf_verbl", "cs_vcrim", "cs_bulge", "cs_other", "ac_incid", "ac_time", "rf_knowl",
#               "ac_stsnd", "ac_other", "sb_hdobj", "sb_outln", "sb_admis", "sb_other", "rf_furt", "rf_bulg", "offverb",
#               "offshld"]

# data = df[cols_to_keep].join(dummy_race.ix[:, :'race_W'])
# data = data.join(dummy_sex.ix[:, :'sex_1'])

data = df[cols_to_keep].join(dummy_race['race_B'])
data = data.join(dummy_sex['sex_0'])
data = data.join(dummy_pct.ix[:, :'pct_122'])


# manually add the intercept
data['intercept'] = 1.0

# define training columns
train_cols = data.columns[1:]

# define model with variable to predict
logit = sm.Logit(data['frisked'], data[train_cols])

# fit the model
result = logit.fit()

print result.summary()