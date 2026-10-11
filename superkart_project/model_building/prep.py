import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("superkart_project/data/SuperKart.csv")
rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {rows_before - len(df)} duplicate rows.")

TARGET = "Product_Store_Sales_Total"
if TARGET not in df:
    raise ValueError(f"Target column {TARGET!r} is missing.")

df = df.dropna(subset=[TARGET])

# Remove columns that are entirely empty, if any
all_null_cols = [c for c in df.columns if df[c].isna().all() and c != TARGET]
if all_null_cols:
    df = df.drop(columns=all_null_cols)
    print("Removed all-null columns:", all_null_cols)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split data (without stratification, as this is a regression task)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Target summary statistics in train:")
print(ytrain.describe())
