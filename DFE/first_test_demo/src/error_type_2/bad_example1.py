import pandas as pd

df = pd.read_excel('./TidyData.xlsx', 'Bad_Example1' )

print(df)


# Rename the first column to 'patient'
df.rename(columns={'Unnamed: 0': 'patient'}, inplace=True)


# Use pd.melt to transform the DataFrame into tidy format.
df_tidy = pd.melt(
    df,
    id_vars=['patient'],                       # Columns to keep (identifier)
    value_vars=['treatmenta', 'treatmentb'],    # Columns to melt into long format
    var_name='trt',                            # New column for treatment type ('treatmenta' or 'treatmentb')
    value_name='result'                        # New column for the treatment result
)

print("\nTidy DataFrame:")
print(df_tidy)
