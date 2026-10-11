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
# Remove columns that are entirely empty, if any; do not drop useful IDs by default.
all_null_cols = [c for c in df.columns if df[c].isna().all() and c != TARGET]
if all_null_cols:
    df = df.drop(columns=all_null_cols)
    print("Removed all-null columns:", all_null_cols)

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# stratify=y keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
