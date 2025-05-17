import pandas as pd

df = pd.read_excel('./TidyData_ErrorType_Two.xlsx', 'Sheet1')
print(df)


# Pivot the DataFrame: use 'machine' and 'date' as index, 'key' for columns, and 'value' as data.
df_tidy = df.pivot(index=['machine', 'date'],   # We use ['machine', 'date'] as the index since each combination of machine and date uniquely identifies a record.
                columns='key', 
                values='value').reset_index()

print("\nTidy DataFrame:")
print(df_tidy)
