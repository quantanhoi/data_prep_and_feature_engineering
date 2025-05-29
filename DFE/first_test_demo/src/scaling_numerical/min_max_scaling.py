'''
Your task: 
    - Open the starter notebook and load housing data
    - Apply a min max scaler to the column total_rooms
    - Verify the transformed column is in range [-1,1]
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------
# 1. read the file
# -------------------------------------------------------------
df = pd.read_csv("housing.csv")

# -------------------------------------------------------------
# 2. scale `total_rooms`
# -------------------------------------------------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
df["total_rooms_scaled"] = scaler.fit_transform(df[["total_rooms"]])

#Verify max and min total_room
max_rooms = df["total_rooms"].max()
min_rooms = df["total_rooms"].min()
print("max total rooms: " + str(max_rooms))
print("min total rooms: " + str(min_rooms))

# -------------------------------------------------------------
# 3. write the result
# -------------------------------------------------------------
df.to_csv("housing_scaled.csv", index=False)   # <- new file with the extra column
print("✅  new file written: housing_scaled.csv")