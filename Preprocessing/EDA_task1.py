from base64 import encode
import numpy as np
import pandas as pd
from functools import reduce

data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")

numerical_features = ['num_beds', 'num_baths', 'size_sqft', 'planning_area']
categorical_features = ['property_type', 'tenure', 'furnishing', 'built_year', 'subzone']
predictor = 'price'


# normalize the numerical features, normalize test data with train statistics
def normalize(train_data, test_data, features, is_test):
    if is_test==True:
        for feature in features:
            mean = np.mean(train_data[feature])
            std = np.std(train_data[feature])
            test_data[feature] = (test_data[feature] - mean) / std
        return test_data
    else:
        for feature in features:
            mean = np.mean(train_data[feature])
            std = np.std(train_data[feature])
            train_data[feature] = (train_data[feature] - mean) / std
        return train_data


def cal_quantile(data):
    Q1 = np.quantile(data, 0.25)
    Q3 = np.quantile(data, 0.75)
    IQR = Q3 - Q1
    floor = Q1 - 1.5 * IQR
    ceil = Q3 + 1.5 * IQR
    return floor, ceil


def encode_and_normal_data(X, is_test):
    # fill beds and baths, delete training data without both
    X['num_beds'] = X.apply(
        lambda x: x['num_beds'] if x['num_beds'] > 0 else X[X['num_baths'] == x['num_baths']]['num_beds'].median(),
        axis=1)
    X['num_baths'] = X.apply(
        lambda x: x['num_baths'] if x['num_baths'] > 0 else X[X['num_beds'] == x['num_beds']]['num_baths'].median(),
        axis=1)
    if is_test:
        X['num_beds'] = X['num_beds'].fillna(X['num_beds'].mean())
        X['num_baths'] = X['num_baths'].fillna(X['num_baths'].mean())
    else:
        X = X.loc[X['num_beds'].notnull()]
        X = X.loc[X['num_baths'].notnull()]

    # discard training data outliers in 'size_sqft' (0, 10000)
    if is_test == False:
        X = X.loc[X['size_sqft'] > 0]
        X = X.loc[X['size_sqft'] < 10000]

    # transform 'property_type'. We only want four types: 'condo', 'hdb', 'house', 'bungalow'.
    # Because most of houses are 'condo', test data with other property_type are considered of 'condo'
    X['property_type'] = X.property_type.str.lower()
    X.loc[X['property_type'].str.contains('hdb'), 'property_type'] = 'hdb'
    X.loc[X['property_type'].str.contains('condo'), 'property_type'] = 'condo'
    X.loc[X['property_type'].str.contains('house'), 'property_type'] = 'house'
    X.loc[X['property_type'].str.contains('bungalow'), 'property_type'] = 'bungalow'
    if is_test:
        X.loc[X['property_type'].str.contains('hdb|condo|house|bungalow') == False, 'property_type'] = 'condo'
    X = X[X['property_type'].str.contains('hdb|condo|house|bungalow')]

    #  transform 'tenure'. We only want two types: 'freehold' and 'leasehold'
    X.loc[X['tenure'].isnull(), 'tenure'] = 'leasehold'
    X.loc[X['tenure'].str.contains('freehold'), 'tenure'] = 'freehold'
    X.loc[X['tenure'].str.contains('leasehold'), 'tenure'] = 'leasehold'
    if is_test:
        X['tenure'] = X['tenure'].fillna(X['tenure'].mode())
    X = X[X['tenure'].str.contains('freehold|leasehold')]

    # transform 'furnishing'. We only want four types: 'unfurnished', 'partial furnished', 'fully furnished', 'unspecified'
    X.loc[X['furnishing'] == 'na', 'furnishing'] = 'unspecified'

    # transform 'built_year'. We only want 5 types: 'before 1990', '1990-2000', '2000-2010', '2010-2020', 'after 2020'
    # X = X.loc[X['built_year'].notnull()]

    # use the mode of subzone to fil the builtyear
    X['built_year'] = X.apply(
        lambda x: x['built_year'] if (x['built_year'] > 0) else (X[X['subzone'] == x['subzone']].mode().iloc[0].iloc[7]), axis=1)
    # use the mode of planningarea to fil the builtyear
    X['built_year'] = X.apply(
        lambda x: x['built_year'] if (x['built_year'] > 0) else (
        X[X['planning_area'] == x['planning_area']].mode().iloc[0].iloc[7]), axis=1)
    # use the mode of alldata to fil the builtyear
    X['built_year'] = X.apply(
        lambda x: x['built_year'] if (x['built_year'] > 0) else (X.mode()).iloc[0].iloc[7], axis=1)

    X['built_year'] = X['built_year'].astype(int)
    X.loc[X['built_year'] <= 1990, 'built_year'] = 1990
    X.loc[(X['built_year'] > 1990) & (X['built_year'] <= 2000), 'built_year'] = 1995
    X.loc[(X['built_year'] > 2000) & (X['built_year'] <= 2010), 'built_year'] = 2005
    X.loc[(X['built_year'] > 2010) & (X['built_year'] <= 2020), 'built_year'] = 2015
    X.loc[X['built_year'] > 2020, 'built_year'] = 2025
    pd.value_counts(X['built_year'])

    # transform 'planning_area'. Calculate the mean price of each area in train data and replace the area name with the mean price
    # for test data without planning_area, replace with the mean of all mean_price
    data_train['planning_area'] = data_train['planning_area'].str.lower()
    mean_price = data_train['price'] / data_train['size_sqft']
    mean_price = pd.concat([data_train['planning_area'], mean_price], axis=1)
    area_mean_price = mean_price.groupby('planning_area').mean()
    area_mean_price.rename(columns={0: 'mean_price'}, inplace=True)
    if is_test:
        X = pd.merge(X, area_mean_price, on='planning_area', how='left')
        X['mean_price'] = X['mean_price'].fillna(X['mean_price'].mean())
    else:
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
    subzone_mall = subzone_mall.loc[:, ['subzone', 'name']].rename(columns={'name': 'mall_count'})

    # find whether there is any mrt stations in each subzone
    mrt_count = mrt.groupby('subzone').count()
    mrt_count = mrt_count.loc[:, 'code']
    subzone_mrt = pd.merge(subzone, mrt_count, on='subzone', how='left')
    subzone_mrt = subzone_mrt.fillna(0)
    subzone_mrt = subzone_mrt.loc[:, ['subzone', 'code']].rename(columns={'code': 'mrt_count'})
    subzone_mrt['mrt_count'] = subzone_mrt['mrt_count'] > 0
    subzone_mrt['mrt_count'] = subzone_mrt['mrt_count'].map({True: 1, False: 0})

    # find whether there is any primary schools in each subzone
    primary_count = primary.groupby('subzone').count()
    primary_count = primary_count.loc[:, 'name']
    subzone_primary = pd.merge(subzone, primary_count, on='subzone', how='left')
    subzone_primary = subzone_primary.fillna(0)
    subzone_primary = subzone_primary.loc[:, ['subzone', 'name']].rename(columns={'name': 'primary_count'})
    subzone_primary['primary_count'] = subzone_primary['primary_count'] > 0
    subzone_primary['primary_count'] = subzone_primary['primary_count'].map({True: 1, False: 0})

    # find whether there is any secondary schools in each subzone
    secondary_count = secondary.groupby('subzone').count()
    secondary_count = secondary_count.loc[:, 'name']
    subzone_secondary = pd.merge(subzone, secondary_count, on='subzone', how='left')
    subzone_secondary = subzone_secondary.fillna(0)
    subzone_secondary = subzone_secondary.loc[:, ['subzone', 'name']].rename(columns={'name': 'secondary_count'})
    subzone_secondary['secondary_count'] = subzone_secondary['secondary_count'] > 0
    subzone_secondary['secondary_count'] = subzone_secondary['secondary_count'].map({True: 1, False: 0})

    # integrate school information (None, primary, secondary or both)
    subzone_primary = pd.merge(subzone_primary, subzone_secondary, on='subzone')
    subzone_primary['school_count'] = subzone_primary['primary_count'] + subzone_primary['secondary_count']
    subzone_primary = subzone_primary.loc[:, ['subzone', 'school_count']]
    subzone_school = subzone_primary.copy()

    # merge all auxiliary results to X, fillna(0)
    merge_table = [subzone_mall, subzone_mrt, subzone_school]
    subzone_auxiliary = reduce(lambda left, right: pd.merge(left, right, on='subzone'), merge_table)
    if is_test:
        X = pd.merge(X, subzone_auxiliary, on='subzone', how='left')
        X = X.fillna(0)
    else:
        X = pd.merge(X, subzone_auxiliary, on='subzone')

    # preprocess categorical data with one-hot encoding
    X.drop('planning_area', axis=1, inplace=True)
    X.drop('subzone', axis=1, inplace=True)
    categoricals = ['property_type', 'tenure', 'built_year', 'furnishing', 'school_count', 'mrt_count']
    X = pd.get_dummies(X, columns=categoricals, drop_first=True)

    # normalize numerical data
    numericals = ['num_beds', 'num_baths', 'size_sqft', 'mean_price']
    # X = normalize(X, numericals)
    return X


def get_test():
    x_train = data_train.loc[:, numerical_features + categorical_features + ['price']]
    x_test = data_test.loc[:, numerical_features + categorical_features]
    x_train = encode_and_normal_data(x_train, is_test=False)
    x_test = encode_and_normal_data(x_test, is_test=True)
    numericals = ['num_beds', 'num_baths', 'size_sqft', 'mean_price']
    x_test = normalize(x_train, x_test, numericals, is_test=True)
    # X = encode_and_normal_data(X, True)
    return x_test


def get_train():
    x_train = data_train.loc[:, numerical_features + categorical_features + ['price']]
    x_test = data_test.loc[:, numerical_features + categorical_features]
    x_train = encode_and_normal_data(x_train, is_test=False)
    x_test = encode_and_normal_data(x_test, is_test=True)
    numericals = ['num_beds', 'num_baths', 'size_sqft', 'mean_price']
    x_train = normalize(x_train, x_test, numericals, is_test=False)
    x_train, y_train = x_train.drop('price', axis=1), x_train['price']
    return x_train, y_train


def save_pred(y_pred, out_file):
    df_y = pd.DataFrame(y_pred)
    df_y.index.name = 'Id'
    df_y.columns = ['Predicted']
    df_y.to_csv(out_file)
