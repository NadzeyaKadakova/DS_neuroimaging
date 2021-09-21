import pandas as pd

#a pandas DataFrame is a fancy dictionary
dict_test   = {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
df_test     = pd.DataFrame(dict_test)
print(dict_test)
print(df_test)

#a more useful example
#create DataFrame object named df1 from an Excel file (.xlsx)
df1     = pd.read_excel('../data/ADNIMERGE_small.xlsx')

#how big is it?
print('Shape: ', df1.shape)

#preview the resulting DataFrame
#or right click on 'df1' in 'Variables' tab and select 'View as DataFrame'
print('*** head ***')
print(df1.head(10))
print('*** tail ***')
print(df1.tail(10))

#print the types in each column
print('Column types')
print(df1.dtypes)

#print the columns, casting them to list first
print('Columns:',list(df1.columns))

#columns are stored as pandas Series objects
print(type(df1['RID']))

#it's a series
print(df1['RID'].head(10))

#get the actual values
print('RID values: ', df1['RID'].to_numpy()[0:30])

#sort by RID, then by years since baseline
df2     = df1.sort_values(by=['RID', 'Years_bl'])
#do the reset in-place and drop the old index
df2.reset_index(drop=True, inplace=True)

#df1 hasn't changed
print(df1.head(10))

#df2 is the sorted version
print(df2.head(10))

#get the first row of the RID column
print(df2['RID'].iloc[0])

#create a 'current age' column
df2['Age_visit'] = df2['AGE'] + df2['Years_bl']
print(df2[['RID', 'AGE', 'Years_bl', 'Age_visit']])

#get the first visit for each subject
df_1st      = df2.drop_duplicates(subset=['RID'])
df_1st.reset_index(drop=True, inplace=True)
print(df_1st[['RID', 'Age_visit', 'DX']])

#get all CN subjects under 75
#notice the parentheses - sometimes 'loc' misbehaves with them
df_1st_filtered  = df_1st.loc[(df_1st.DX=='CN') & (df_1st.Age_visit < 75)]
print(df_1st_filtered[['RID', 'Age_visit', 'DX']])


#get all MCI subjects

print('done')