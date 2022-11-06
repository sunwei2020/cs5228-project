import pandas as pd
import numpy as np

numerical_features = ['num_beds', 'num_baths', 'size_sqft', 'planning_area', 'price']
categorical_features = ['property_type', 'built_year']


def normalize(train_data, features):
    for feature in features:
        mean = np.mean(train_data[feature])
        std = np.std(train_data[feature])
        train_data[feature] = (train_data[feature] - mean) / std
    return train_data

def data_processing(df_sample):
    df = df_sample.copy()
    df = df.loc[:, ['listing_id']+numerical_features + categorical_features]
    df = df.dropna()
    
    df['property_type'] = df.property_type.str.lower()
    df.loc[df['property_type'].str.contains('hdb'), 'property_type'] = 'hdb'
    df.loc[df['property_type'].str.contains('condo'), 'property_type'] = 'condo'
    df.loc[df['property_type'].str.contains('house'), 'property_type'] = 'house'
    df.loc[df['property_type'].str.contains('bungalow'), 'property_type'] = 'bungalow'
    df = df[df['property_type'].str.contains('hdb|condo|house|bungalow')]
    
    df['built_year'] = df['built_year'].astype(int)
    df.loc[df['built_year'] <= 1990, 'built_year'] = 1990
    df.loc[(df['built_year'] > 1990) & (df['built_year'] <= 2000), 'built_year'] = 1995
    df.loc[(df['built_year'] > 2000) & (df['built_year'] <= 2010), 'built_year'] = 2005
    df.loc[(df['built_year'] > 2010) & (df['built_year'] <= 2020), 'built_year'] = 2015
    df.loc[df['built_year'] > 2020, 'built_year'] = 2025
    pd.value_counts(df['built_year'])
    
    df['planning_area'] = df['planning_area'].str.lower()
    mean_price = df['price'] / df['size_sqft']
    mean_price = pd.concat([df['planning_area'], mean_price], axis=1)
    area_mean_price = mean_price.groupby('planning_area').mean()
    area_mean_price.rename(columns={0: 'area_mean_price'}, inplace=True)
    df = pd.merge(df, area_mean_price, on='planning_area')
    
    df.loc[df['price'] <= 1000000, 'price'] = 1
    df.loc[(df['price'] > 1000000) & (df['price'] <= 2000000), 'price'] = 2
    df.loc[(df['price'] > 2000000) & (df['price'] <= 3000000), 'price'] = 3
    df.loc[df['price'] > 3000000, 'price'] = 4
    
    numerical = ['num_beds', 'num_baths', 'size_sqft', 'area_mean_price']
    categorical = ['property_type', 'built_year', 'price']
    df = pd.get_dummies(df, columns=categorical, drop_first=True)
    df = normalize(df, numerical)
    df.drop('planning_area', axis=1, inplace=True)
    
    return df

    
    
