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


# Create dummy columns
dummy_id = pd.get_dummies(df['typeofid'], prefix='typeofid')
dummy_trhsloc = pd.get_dummies(df['trhsloc'], prefix='trhsloc')
dummy_race = pd.get_dummies(df['race'], prefix='race')
dummy_sex = pd.get_dummies(df['sex'], prefix='sex')
dummy_pct = pd.get_dummies(df['pct'], prefix='pct')
dummy_haircolr = pd.get_dummies(df['haircolr'], prefix='haircolr')
dummy_eyecolor = pd.get_dummies(df['eyecolor'], prefix='eyecolor')
dummy_build = pd.get_dummies(df['build'], prefix='build')


# Merge dummy columns
data = df.join(dummy_id)
data = data.join(dummy_trhsloc)
data = data.join(dummy_race)
data = data.join(dummy_sex)
data = data.join(dummy_pct)
data = data.join(dummy_haircolr)
data = data.join(dummy_eyecolor)
data = data.join(dummy_build)

# Create CSV
data.to_csv('data_with_dummies.csv')



