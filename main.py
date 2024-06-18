import pandas as pd
from functools import reduce
import json
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


########################## Preprocessing data for right form ##########################
# Read json as dict
with open('output_clean_date_technical.json', 'r') as file:
    json_file = json.loads(file.read())

# Combine 5 dataframe into 1
# -------------------------------
# | financialGrowth             |
# | ratios                      |
# | cashFlowStatementGrowth     |
# | incomeStatementGrowth       |
# | balanceSheetStatementGrowth |
# -------------------------------
# Change type into dataframe
temp_info = [pd.DataFrame(json_file[key]) for key in 
             ['balanceSheetStatementGrowth', 'cashFlowStatementGrowth', 'incomeStatementGrowth', 'financialGrowth', 'ratios']]
# Merge 5 dataframe into 1
df_merged_info = reduce(lambda  left,right: pd.merge(left,right,on=["date", "symbol", "calendarYear", "period"], how='outer'), temp_info)


# Combine 5 dataframe into 1
# -------------------------------
# | historicalPriceFull         |
# | tech5                       |
# | tech20                      |
# | tech60                      |
# | tech252                     |
# -------------------------------
# Make list to fill every tech file
temp_tech = []
for every_tech_name in ['tech5', 'tech20', 'tech60', 'tech252']:
    # To dataframe
    temp_df = pd.DataFrame(json_file[every_tech_name])
    # To add suffix, because columns name duplicate
    temp_df.columns = temp_df.columns.map(lambda x : x + f'_{every_tech_name}' if x not in 
                                          ['date', 'open', 'high', 'low', 'close' ,'volume'] else x)
    # Change type of [date] column to datetime in every tech df
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    temp_tech.append(temp_df)

# Change type in [historicalPrice]
transformed_historicalPrice = pd.DataFrame([dict(item, **{'symbol':'1101.TW'}) for item in json_file['historicalPriceFull']['historical']])
# Change type of [date] column to datetime in [historicalPrice]
transformed_historicalPrice['date'] = pd.to_datetime(transformed_historicalPrice['date']) 
# Append to list
temp_tech.append(transformed_historicalPrice)

# Merge 5 dataframe into 1
df_merged_price = reduce(lambda  left,right: pd.merge(left,right,on=['date', 'open', 'high', 'low' ,'close', 'volume'], 
                                                      how='outer'), temp_tech)

# ----------------------------------
# -- Merge (df_merged_info + df_merged_price) on Q1, Q2, Q3, Q4 -- #
## Change columns name
df_merged_info.rename(columns = {'date':'date_info'}, inplace = True)
df_merged_price.rename(columns = {'date':'date_price'}, inplace = True)


# -- Merge 2 dataframe -- #
## Function to map year and quarter to a start date
def map_quarter_to_dates(row):
    year = int(row['calendarYear'])
    quarter = row['period']
    if quarter == 'Q1':
        return pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-03-31')
    elif quarter == 'Q2':
        return pd.Timestamp(f'{year}-04-01'), pd.Timestamp(f'{year}-06-30')
    elif quarter == 'Q3':
        return pd.Timestamp(f'{year}-07-01'), pd.Timestamp(f'{year}-09-30')
    elif quarter == 'Q4':
        return pd.Timestamp(f'{year}-10-01'), pd.Timestamp(f'{year}-12-31')

## Apply the function to create the new column
df_merged_info[['Start_Date', 'End_Date']] = df_merged_info.apply(map_quarter_to_dates, axis=1, result_type='expand')

## Sort before merge_asof
df_merged_price = df_merged_price.sort_values(by='date_price')
df_merged_info = df_merged_info.sort_values(by='Start_Date')

## Merge 2 dataframe based on date
df_answer = pd.merge_asof(df_merged_price, df_merged_info, left_on='date_price', right_on='Start_Date')


# -- Eliminate date not in correct place -- #
## Create a mask for rows where date_price is outside the start and end dates
mask = (df_answer['date_price'] < df_answer['Start_Date']) | (df_answer['date_price'] > df_answer['End_Date'])

## Fill the specified columns with NaN where the mask is True
df_answer.loc[mask, df_merged_info.columns] = pd.NA


# -- Fix missing data from df_merged_info_dataframe -- #
## Find which row is not added in dataframe_df_merged_info
elements_not_been_added = set(df_merged_info['Start_Date'].to_list()) - set(df_answer['Start_Date'].to_list())
rows_not_been_added = df_merged_info[df_merged_info['Start_Date'].isin(elements_not_been_added)]

## Concat rows_not_been_added into dataframe
df_answer = pd.concat([df_answer, rows_not_been_added],axis=0)

## Delete redundant columns
df_answer = df_answer.drop(['Start_Date', 'End_Date'], axis=1)

# ----------------------------------
# Store to csv
df_answer.to_csv('data_to_csv.csv', sep=',', index=False)




########################## Preprocessing data for model ##########################
from sklearn.preprocessing import MinMaxScaler
df_preprocessed = df_answer.copy()

# Drop can't minmax scalar columns
df_preprocessed = df_preprocessed.drop(['symbol','symbol_x','date_price'], axis=1)

# Get columns for minmax scalar
df_preprocessed = df_preprocessed[df_preprocessed.describe().columns]

# MinMaxscalar
min_max = MinMaxScaler()
df_preprocessed = pd.DataFrame(min_max.fit_transform(df_preprocessed), columns=df_preprocessed.keys())

# Drop rows contains Nan
df_preprocessed.dropna()

# Store to csv
df_preprocessed.to_csv('data_to_csv_preprocessed.csv', sep=',', index=False)