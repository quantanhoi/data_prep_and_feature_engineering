import pandas as pd

df = pd.read_json('./yelp_academic_dataset_business.json', lines = True)
# Binarize the review_count column
print("median of review count: " + str(df['review_count'].median()))
# Using qcut with 2 bins (median-based binarization)
df['review_count_binarized'] = pd.qcut(df['review_count'], 
                                       q=2, 
                                       labels=[0, 1])
df.to_csv('yelp_binarized.csv', index = False)


