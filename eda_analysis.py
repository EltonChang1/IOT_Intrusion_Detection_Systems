import pandas as pd
import numpy as np

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("=" * 80)
print("LOADING DATASETS")
print("=" * 80)

# Load datasets
df1 = pd.read_csv('BotNeTIoT-L01_label_NoDuplicates.csv')
df2 = pd.read_csv('BoTNeTIoT-L01-v2.csv')

print("\n" + "=" * 80)
print("DATASET 1: BotNeTIoT-L01_label_NoDuplicates.csv")
print("=" * 80)

print("\n1. Basic Information:")
print(f"   - Shape: {df1.shape}")
print(f"   - Memory Usage: {df1.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n   Columns ({len(df1.columns)}):")
print(f"   {df1.columns.tolist()}")

print("\n2. Data Types:")
print(df1.dtypes)

print("\n3. Missing Values:")
missing1 = df1.isnull().sum()
if missing1.sum() > 0:
    print(missing1[missing1 > 0])
else:
    print("   No missing values found!")

print("\n4. First few rows:")
print(df1.head())

print("\n5. Statistical Summary:")
print(df1.describe())

print("\n6. Class Distribution (label):")
if 'label' in df1.columns:
    print(df1['label'].value_counts().sort_index())
    print(f"\n   Class Balance:")
    print(df1['label'].value_counts(normalize=True).sort_index() * 100)

print("\n7. Duplicate Rows:")
print(f"   Number of duplicates: {df1.duplicated().sum()}")

print("\n\n" + "=" * 80)
print("DATASET 2: BoTNeTIoT-L01-v2.csv")
print("=" * 80)

print("\n1. Basic Information:")
print(f"   - Shape: {df2.shape}")
print(f"   - Memory Usage: {df2.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n   Columns ({len(df2.columns)}):")
print(f"   {df2.columns.tolist()}")

print("\n2. Data Types:")
print(df2.dtypes)

print("\n3. Missing Values:")
missing2 = df2.isnull().sum()
if missing2.sum() > 0:
    print(missing2[missing2 > 0])
else:
    print("   No missing values found!")

print("\n4. First few rows:")
print(df2.head())

print("\n5. Statistical Summary:")
numeric_cols = df2.select_dtypes(include=[np.number]).columns
print(df2[numeric_cols].describe())

print("\n6. Class Distribution (label):")
if 'label' in df2.columns:
    print(df2['label'].value_counts().sort_index())
    print(f"\n   Class Balance:")
    print(df2['label'].value_counts(normalize=True).sort_index() * 100)

print("\n7. Categorical Columns Analysis:")
categorical_cols = df2.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n   {col}:")
    print(df2[col].value_counts())

print("\n8. Duplicate Rows:")
print(f"   Number of duplicates: {df2.duplicated().sum()}")

print("\n\n" + "=" * 80)
print("COMPARISON BETWEEN DATASETS")
print("=" * 80)

print("\n1. Common Columns:")
common_cols = set(df1.columns) & set(df2.columns)
print(f"   Number of common columns: {len(common_cols)}")
print(f"   {sorted(common_cols)}")

print("\n2. Unique to Dataset 1:")
unique1 = set(df1.columns) - set(df2.columns)
if unique1:
    print(f"   {sorted(unique1)}")
else:
    print("   None")

print("\n3. Unique to Dataset 2:")
unique2 = set(df2.columns) - set(df1.columns)
if unique2:
    print(f"   {sorted(unique2)}")
else:
    print("   None")

print("\n4. Size Comparison:")
print(f"   Dataset 1: {df1.shape[0]:,} rows")
print(f"   Dataset 2: {df2.shape[0]:,} rows")
print(f"   Difference: {abs(df1.shape[0] - df2.shape[0]):,} rows")

print("\n\n" + "=" * 80)
print("FEATURE ANALYSIS")
print("=" * 80)

# Get numeric columns for dataset 1
numeric_df1 = df1.select_dtypes(include=[np.number])
if 'label' in numeric_df1.columns:
    features_df1 = numeric_df1.drop('label', axis=1)
else:
    features_df1 = numeric_df1

print("\n1. Features with Zero Variance:")
zero_var = features_df1.columns[features_df1.var() == 0]
if len(zero_var) > 0:
    print(f"   {list(zero_var)}")
else:
    print("   None")

print("\n2. Features with Very Low Variance (< 0.01):")
low_var = features_df1.columns[(features_df1.var() > 0) & (features_df1.var() < 0.01)]
if len(low_var) > 0:
    print(f"   {list(low_var)}")
else:
    print("   None")

print("\n3. Feature Value Ranges (Min, Max):")
for col in features_df1.columns[:10]:  # Show first 10
    print(f"   {col}: [{features_df1[col].min():.2e}, {features_df1[col].max():.2e}]")

print("\n4. Skewness Analysis (|skew| > 1):")
skewness = features_df1.skew()
highly_skewed = skewness[abs(skewness) > 1].sort_values(ascending=False)
if len(highly_skewed) > 0:
    print(highly_skewed.head(10))
else:
    print("   No highly skewed features")

print("\n\n" + "=" * 80)
print("CORRELATION ANALYSIS (Dataset 1)")
print("=" * 80)

# Calculate correlation matrix
corr_matrix = features_df1.corr()

print("\n1. Highly Correlated Feature Pairs (|corr| > 0.9):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                corr_matrix.columns[i], 
                corr_matrix.columns[j], 
                corr_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:20]:
        print(f"   {feat1} <-> {feat2}: {corr:.4f}")
else:
    print("   No highly correlated pairs found")

print("\n\n" + "=" * 80)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 80)

if 'label' in df1.columns:
    print("\nDataset 1:")
    print(f"   Total samples: {len(df1):,}")
    print(f"   Attack samples (0): {(df1['label'] == 0).sum():,} ({(df1['label'] == 0).sum() / len(df1) * 100:.2f}%)")
    print(f"   Normal samples (1): {(df1['label'] == 1).sum():,} ({(df1['label'] == 1).sum() / len(df1) * 100:.2f}%)")

if 'label' in df2.columns:
    print("\nDataset 2:")
    print(f"   Total samples: {len(df2):,}")
    print(f"   Attack samples (0): {(df2['label'] == 0).sum():,} ({(df2['label'] == 0).sum() / len(df2) * 100:.2f}%)")
    print(f"   Normal samples (1): {(df2['label'] == 1).sum():,} ({(df2['label'] == 1).sum() / len(df2) * 100:.2f}%)")
    
    if 'Attack' in df2.columns:
        print("\n   Attack Type Distribution:")
        print(df2['Attack'].value_counts())
    
    if 'Attack_subType' in df2.columns:
        print("\n   Attack SubType Distribution:")
        print(df2['Attack_subType'].value_counts())
    
    if 'Device_Name' in df2.columns:
        print("\n   Device Distribution:")
        print(df2['Device_Name'].value_counts())

print("\n\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
