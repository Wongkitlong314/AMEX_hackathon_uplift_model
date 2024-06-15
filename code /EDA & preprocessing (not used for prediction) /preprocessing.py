import numpy as np
from scipy.stats.mstats import winsorize
import pandas as pd
import random

seed = 1

# standardize feature 
# 1. create df of different dimension: 
# - merchant
# - customer
# - customer & merchant
# - customer & industry
# 2. use the entire training data to calculate the mean and std 
# 3. use the calculate mean and std for each feature to standardize the required dataset. 

# create dim df
def create_customer_df(df):
    # Specify the columns to be dropped
    columns_to_drop = [col for col in df.columns if col.startswith("merchant") or 
                                                  col.startswith("customer_industry_spend") or 
                                                  col.startswith("distance") or 
                                                  col.startswith("customer_merchant_") or  
                                                  (col.startswith("customer_spend_") and int(col.split("_")[-1]) in [4] + list(range(13, 20)))]

#     # Group by 'customer' and perform aggregations
#     aggregations = {
#         'ind_recommended': 'sum',
#         'activation': 'sum'}
    aggregations = {}

    # Add mean aggregation for all other columns
    for col in df.columns:
        if col not in aggregations and col not in columns_to_drop:
            aggregations[col] = 'mean'

    # Perform aggregation directly without creating an intermediate DataFrame
    customer_df = df.drop(columns=columns_to_drop).groupby('customer', as_index = False).agg(aggregations)

    return customer_df 

def create_customer_industry_df(df):
    # Specify the columns to be dropped
    column_list = [col for col in df.columns if  col.startswith("customer_spend_04") or 
                                                  col.startswith("customer_industry_spend_") or 
                                                  (col.startswith("customer_spend_") and int(col.split("_")[-1]) in [4] + list(range(13, 20)))] 

    column_list.append('merchant_profile_01') 
    
    aggregations = {}
    # Add mean aggregation for all other columns
    for col in df.columns:
        if col in column_list:
            aggregations[col] = 'mean'
    
    # Perform aggregation directly without creating an intermediate DataFrame
    df = df.groupby(['merchant_profile_01', 'customer'], as_index = False).agg(aggregations)[column_list]
    df.fillna(0.0, inplace=True)
    mean_df = df.groupby('merchant_profile_01', as_index = False).mean()
    std_df = df.groupby('merchant_profile_01', as_index = False).std(ddof = 1)
    
    return mean_df, std_df 

def create_merchant_df(df):
    # Specify the columns to be dropped
    column_list = [col for col in df.columns if  col.startswith("merchant_spend")] 

    column_list.extend(["merchant", "merchant_profile_02", "merchant_profile_03"])
    
    aggregations = {}
    # Add mean aggregation for all other columns
    for col in df.columns:
        if col in column_list:
            aggregations[col] = 'mean'

    # Perform aggregation directly without creating an intermediate DataFrame
    df = df.groupby('merchant', as_index = False).agg(aggregations)[column_list]
    df.fillna(0.0, inplace=True)

    return df 

def create_customer_merchant_df(df):
    column_list = [col for col in df.columns if col.startswith("customer_merchant_") or 
                                                  col.startswith("distance_") ] 
    return df[column_list]

def dim_std_mean(describe_df, data):
    df = data.copy()
    # drop col
    columns_to_drop = ['ind_recommended', 'activation', 'customer', 'merchant']
    drop_columns = [col for col in describe_df.columns if col in columns_to_drop ]
    describe_df.drop(drop_columns, axis=1, inplace=True)
    
    # Extract mean and std from the describe DataFrame
    means = describe_df.iloc[1]
    stds = describe_df.iloc[2]

    # Find overlapping columns
    overlapping_columns = means.index.intersection(df.columns)

    # Standardize overlapping columns
    for col in overlapping_columns:
        df[col] = (df[col] - means[col]) / stds[col]  # Standardize column in df

    return df

def dim_std_mean_industry(mean_df, std_df, df): 
    # dim_df = winsorization(dim_df)

    # Calculate mean and standard deviation for each column grouped by 'merchant_profile_01'
    # mean_df = dim_df.groupby('merchant_profile_01', as_index = False).mean()
    # std_df = dim_df.groupby('merchant_profile_01', as_index = False).std(ddof = 1)
    
    # Drop specified columns if they exist in mean and std DataFrames
    columns_to_drop = ['ind_recommended', 'activation', 'customer', 'merchant']
    drop_columns = [col for col in mean_df.columns if col in columns_to_drop ]
    mean_df.drop(drop_columns, axis=1, inplace=True)
    std_df.drop(drop_columns, axis=1, inplace=True)

    # Join df with mean and std DataFrames on 'merchant_profile_01'
    df = df.merge(mean_df, how='left', on='merchant_profile_01', suffixes=('', '_mean'))
    df = df.merge(std_df, how='left', on='merchant_profile_01', suffixes=('', '_std'))

    col_list = list(mean_df.columns)
    col_list.remove('merchant_profile_01')
    # Standardize columns in df using mean and std columns
    for col in col_list:
        mean_col = col + '_mean'
        std_col = col + '_std'
        df[col] = (df[col] - df[mean_col]) / df[std_col]
        df.drop([mean_col, std_col], axis=1, inplace=True)
    
    return df

def data_preprocessing(customer_df, customer_merchant_df, merchant_df, customer_industry_mean_df, customer_industry_std_df, stand_df, batch_num=10):
    # stand_df = data_to_standardize.copy()
    total_rows = stand_df.shape[0]
    start_index = 0
    batch_size = int(total_rows / batch_num)

    while start_index < total_rows:
        end_index = min(start_index + batch_size, total_rows)
        batch_df = stand_df.iloc[start_index:end_index]

        batch_df = dim_std_mean(customer_df, batch_df)
        batch_df = dim_std_mean(customer_merchant_df, batch_df)
        batch_df = dim_std_mean(merchant_df, batch_df)
        batch_df = dim_std_mean_industry(customer_industry_mean_df, customer_industry_std_df, batch_df)

        stand_df.iloc[start_index:end_index] = batch_df
        start_index += batch_size
    

    return stand_df
