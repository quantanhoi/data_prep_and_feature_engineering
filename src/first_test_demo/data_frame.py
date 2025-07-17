import pandas as pd

df = pd.DataFrame({'col': [0, 3, 2 , 4, 5, 10]})
# print(type(df['col']))
# print(type(df[['col']]))

# print(df.head(n=1))
# print(df.tail())
# print(df.describe())
# print(df.index)
# print(df<=2)
# print(df[df<=2])
# df2 = pd.DataFrame({'neg_col': [-3, -5, -9]})
# new_df = pd.concat([df, df2], axis=0)

# print(new_df)



df = pd.DataFrame({
    'City': ['Paris', 'London', 'Berlin', 'London'],
    'Population': [2148271, 8982000, 3645000, 12345678]
})

pop_london = df.loc[df['City'] == 'London', 'Population']
print(pop_london)
