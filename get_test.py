import numpy as np
import pandas as pd
from functools import reduce

data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")
data_test.head()

numerical_features = ['num_beds', 'num_baths', 'size_sqft', 'planning_area']
categorical_features = ['property_type', 'tenure', 'furnishing', 'built_year', 'subzone']
predictor = 'price'


# normalize the numerical features
def normalize(data, features):
    for feature in features:
        mean = np.mean(data[feature])
        std = np.std(data[feature])
        data[feature] = (data[feature] - mean) / std
    return data


def main():
    X = data_test.loc[:, numerical_features+categorical_features]

    # discard null values
    X = X.loc[X['num_beds'].notnull()]
    X = X.loc[X['num_baths'].notnull()]

    # discard outliers in 'size_sqft' (0, 10000)
    X = X.loc[X['size_sqft'] > 0]
    X = X.loc[X['size_sqft'] < 10000]

    # transform 'property_type'. We only want four types: 'condo', 'hdb', 'house', 'bungalow'
    X['property_type'] = X.property_type.str.lower()
    X.loc[X['property_type'].str.contains('hdb'), 'property_type'] = 'hdb'
    X.loc[X['property_type'].str.contains('condo'), 'property_type'] = 'condo'
    X.loc[X['property_type'].str.contains('house'), 'property_type'] = 'house'
    X.loc[X['property_type'].str.contains('bungalow'), 'property_type'] = 'bungalow'
    X = X[X['property_type'].str.contains('hdb|condo|house|bungalow')]

    #  transform 'tenure'. We only want two types: 'freehold' and 'leasehold'
    X.loc[X['tenure'].isnull(), 'tenure'] = 'leasehold'
    X.loc[X['tenure'].str.contains('freehold'), 'tenure'] = 'freehold'
    X.loc[X['tenure'].str.contains('leasehold'), 'tenure'] = 'leasehold'
    X = X[X['tenure'].str.contains('freehold|leasehold')]

    # transform 'furnishing'. We only want four types: 'unfurnished', 'partial furnished', 'fully furnished', 'unspecified'
    X.loc[X['furnishing']=='na', 'furnishing'] = 'unspecified'

    # transform 'built_year'. We only want 5 types: 'before 1990', '1990-2000', '2000-2010', '2010-2020', 'after 2020'
    X = X.loc[X['built_year'].notnull()]
    X['built_year'] = X['built_year'].astype(int)
    X.loc[X['built_year'] <= 1990, 'built_year'] = 1990
    X.loc[(X['built_year'] > 1990) & (X['built_year'] <= 2000), 'built_year'] = 1995
    X.loc[(X['built_year'] > 2000) & (X['built_year'] <= 2010), 'built_year'] = 2005
    X.loc[(X['built_year'] > 2010) & (X['built_year'] <= 2020), 'built_year'] = 2015
    X.loc[X['built_year'] > 2020, 'built_year'] = 2025
    pd.value_counts(X['built_year'])

    data_train['planning_area'] = data_train['planning_area'].str.lower()
    mean_price = data_train['price']/data_train['size_sqft']
    mean_price = pd.concat([data_train['planning_area'], mean_price], axis=1)
    area_mean_price = mean_price.groupby('planning_area').median()
    area_mean_price.rename(columns={0:'mean_price'}, inplace=True)
    X = pd.merge(X, area_mean_price, on='planning_area')

    # read auxiliary data
    shopping_mall = "data/auxiliary-data/sg-shopping-malls.csv"
    mrt = "data/auxiliary-data/sg-mrt-stations.csv"
    subzone = "data/auxiliary-data/sg-subzones.csv"
    primary = "data/auxiliary-data/sg-primary-schools.csv"
    secondary = "data/auxiliary-data/sg-secondary-schools.csv"
    shopping_mall = pd.read_csv(shopping_mall)
    mrt = pd.read_csv(mrt)
    subzone = pd.read_csv(subzone)
    primary = pd.read_csv(primary)
    secondary = pd.read_csv(secondary)

    # pick up name of subzones
    subzone = subzone['name']
    copy = subzone.copy()
    copy.loc[:] = 0
    subzone = pd.merge(subzone, copy, left_index=True, right_index=True)
    subzone.columns = ['subzone', 'count']

    # count number of shopping malls in each subzone
    mall_count = shopping_mall.groupby('subzone').count()
    mall_count = mall_count.loc[:, 'name']
    subzone_mall = pd.merge(subzone, mall_count, on='subzone', how='left')
    subzone_mall = subzone_mall.fillna(0)
    subzone_mall = subzone_mall.loc[:,['subzone','name']].rename(columns={'name':'mall_count'})

    # find whether there is any mrt stations in each subzone
    mrt_count = mrt.groupby('subzone').count()
    mrt_count = mrt_count.loc[:, 'code']
    subzone_mrt = pd.merge(subzone, mrt_count, on='subzone', how='left')
    subzone_mrt = subzone_mrt.fillna(0)
    subzone_mrt = subzone_mrt.loc[:,['subzone','code']].rename(columns={'code':'mrt_count'})
    subzone_mrt['mrt_count'] = subzone_mrt['mrt_count']>0
    subzone_mrt['mrt_count'] = subzone_mrt['mrt_count'].map({True:1, False:0})

    # find whether there is any primary schools in each subzone
    primary_count = primary.groupby('subzone').count()
    primary_count = primary_count.loc[:, 'name']
    subzone_primary = pd.merge(subzone, primary_count, on='subzone', how='left')
    subzone_primary = subzone_primary.fillna(0)
    subzone_primary = subzone_primary.loc[:,['subzone','name']].rename(columns={'name':'primary_count'})
    subzone_primary['primary_count'] = subzone_primary['primary_count']>0
    subzone_primary['primary_count'] = subzone_primary['primary_count'].map({True:1, False:0})

    # find whether there is any secondary schools in each subzone
    secondary_count = secondary.groupby('subzone').count()
    secondary_count = secondary_count.loc[:, 'name']
    subzone_secondary = pd.merge(subzone, secondary_count, on='subzone', how='left')
    subzone_secondary = subzone_secondary.fillna(0)
    subzone_secondary = subzone_secondary.loc[:,['subzone','name']].rename(columns={'name':'secondary_count'})
    subzone_secondary['secondary_count'] = subzone_secondary['secondary_count']>0
    subzone_secondary['secondary_count'] = subzone_secondary['secondary_count'].map({True:1, False:0})

    # integrate school information (None, primary, secondary or both)
    subzone_primary = pd.merge(subzone_primary, subzone_secondary, on='subzone')
    subzone_primary['school_count'] = subzone_primary['primary_count'] + subzone_primary['secondary_count']
    subzone_primary = subzone_primary.loc[:,['subzone','school_count']]
    subzone_school = subzone_primary.copy()

    # merge all auxiliary results to X
    merge_table = [subzone_mall, subzone_mrt, subzone_school]
    subzone_auxiliary = reduce(lambda left,right: pd.merge(left,right,on='subzone'), merge_table)
    X = pd.merge(X, subzone_auxiliary, on='subzone')

    # preprocess categorical data with one-hot encoding
    X.drop('planning_area', axis=1, inplace=True)
    X.drop('subzone', axis=1, inplace=True)
    categoricals = ['property_type', 'tenure', 'built_year', 'furnishing', 'school_count', 'mrt_count']
    X = pd.get_dummies(X, columns=categoricals, drop_first=True)

    # normalize numerical data
    numericals = ['num_beds', 'num_baths', 'size_sqft', 'mean_price']
    X = normalize(X, numericals)
    X.rename(columns={'mean_price':'planning_area'}, inplace=True)

    x_test = X
    return x_test
