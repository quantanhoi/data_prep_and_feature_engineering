import pandas as pd

df = pd.read_csv('./transaction_data.csv')
print(df)


df['TotalAmount'] = df['NumberOfItemsPurchased'] * df['CostPerItem']
print("DataFrame with TotalAmount:")
print(df)



df_grouped = df.groupby('Country').agg({
    'NumberOfItemsPurchased': 'sum',  # Sum the items across transactions
    'CostPerItem': 'mean',              # Compute the mean cost per item
    'TotalAmount': 'sum'
}).reset_index()

print(df_grouped)