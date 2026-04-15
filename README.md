# IoT Botnet Attack Detection

**Team Members:** Jinhong Lin, Elton Chang, Shiwei Jiang

## Overview

As Internet of Things (IoT) devices proliferate in homes, offices, and critical infrastructure, they have become prime targets for malware and botnet attacks. The Mirai and Gafgyt botnets have demonstrated the devastating potential of compromised IoT devices—turning ordinary cameras, routers, and smart home devices into powerful attack vectors for distributed denial-of-service (DDoS) attacks.

The challenge lies in **accurately distinguishing between benign IoT traffic and malicious botnet activity in real-time**, enabling network administrators to rapidly detect and isolate compromised devices before they can be weaponized. This repository contains code, analysis, and experiments for developing and validating machine learning and deep learning models to detect and classify IoT botnet attacks.

## Problem Statement & Research Questions

**Primary Question:** How can we accurately detect and classify IoT botnet attacks (Mirai and Gafgyt) in real-time network traffic using machine learning models trained on network flow statistics?

**Specific Research Questions:**
1. **Binary Classification**: Can machine learning models reliably distinguish between benign IoT traffic and malicious botnet activity with high accuracy, precision, and recall?
2. **Attack Type Identification**: Can we identify the specific botnet family (Mirai vs. Gafgyt) or attack subtype (UDP, TCP, SYN, etc.) from network traffic patterns alone?
3. **Feature Importance**: Which network flow statistics are most discriminative for detecting botnet behavior, and can we reduce the feature space while maintaining detection performance?
4. **Device-Specific Detection**: Do attack signatures vary across different IoT device types, and should device-specific models be trained for better detection accuracy?
5. **Practical Deployment**: What is the optimal balance between model complexity, inference latency, and detection accuracy for deployment in real-time IDS systems?

**Ultimate Goal:** Proactive defense—catching compromised devices *before* they can participate in large-scale attacks like the 2016 Mirai DDoS that took down major websites.

## Dataset

This project uses the **BoTNeTIoT-L01** IoT intrusion detection dataset from Kaggle. The dataset is large and suitable for machine learning/deep learning experiments:

- **Cleaned Version** (`BotNeTIoT-L01_label_NoDuplicates.csv`): ~2.43 million rows, 25 columns (~463 MB)
- **Extended Version** (`BoTNeTIoT-L01-v2.csv`): ~7.06 million rows, 27 columns (~2.7 GB) with metadata (device name, attack family, attack subtype)

The BoTNeTIoT-L01 dataset is derived from the original detection_of_IoT_botnet_attacks_N_BaIoT dataset, with reduced redundancy by selecting features from a 10-second time window only. It provides a robust benchmark with over 9.3 million labeled network flow samples capturing authentic IoT device behavior alongside multiple botnet attack variants, making it ideal for developing and validating next-generation IoT intrusion detection systems.

## Dataset Files

### 1. BotNeTIoT-L01_label_NoDuplicates.csv
- **Rows:** 2,426,574 samples
- **Columns:** 25 features (23 engineered features + 1 index + 1 label)
- **Size:** ~463 MB
- **Duplicates:** 0 (removed)
- **Description:** Cleaned version with no duplicate samples

### 2. BoTNeTIoT-L01-v2.csv
- **Rows:** 7,062,606 samples
- **Columns:** 27 features (23 engineered features + 3 metadata + 1 label)
- **Size:** ~2.7 GB
- **Duplicates:** 621,659 duplicates present
- **Description:** Extended version with device names and attack type information

## Dataset Characteristics

### Data Collection
The dataset contains traffic from **9 IoT devices** sniffed using **Wireshark** in a local network environment with a central switch. All traffic was captured from `.pcap` files and processed to extract statistical features.

### IoT Devices Included
1. **Philips B120N10 Baby Monitor** (1,098,677 samples)
2. **Danmini Doorbell** (1,018,298 samples)
3. **SimpleHome XCS7-1002 Security Camera** (863,056 samples)
4. **SimpleHome XCS7-1003 Security Camera** (850,826 samples)
5. **Provision PT-838 Security Camera** (836,891 samples)
6. **Ecobee Thermostat** (835,876 samples)
7. **Provision PT-737E Security Camera** (828,260 samples)
8. **Samsung SNH-1011-N Webcam** (375,222 samples)
9. **Ennio Doorbell** (355,500 samples)

### Attack Types

The dataset includes two major botnet families with multiple attack variants:

#### Mirai Botnet (3,668,402 samples)
- UDP flood attacks
- TCP attacks
- Network scanning
- SYN flood
- ACK flood
- UDP plain

#### Gafgyt Botnet (2,838,272 samples)
- Combo attacks
- TCP attacks
- Junk attacks
- UDP attacks
- Scanning

### Attack Subtypes Distribution
| Attack Subtype | Count | Percentage |
|---------------|-------|-----------|
| UDP | 2,176,365 | 30.81% |
| TCP | 859,850 | 12.17% |
| Scan | 793,090 | 11.23% |
| SYN | 733,299 | 10.38% |
| ACK | 643,821 | 9.11% |
| Normal | 555,932 | 7.87% |
| UDPPlain | 523,304 | 7.41% |
| Combo | 515,156 | 7.29% |
| Junk | 261,789 | 3.71% |

## Feature Engineering

The dataset contains **23 statistically engineered features** extracted using a **10-second time window** with a **decay factor of 0.1** (referred to as L0.1 in the literature).

### Base Packet Features (4)
Four fundamental features were extracted from packet capture files:
1. **Packet count**
2. **Jitter** (packet delay variation)
3. **Size of outbound packets only**
4. **Size of outbound and inbound packets together**

### Statistical Measures (7)
For each base feature, the following statistical measures were computed:
1. **Mean** - Average value
2. **Variance** - Spread of values
3. **Count/Weight** - Number of packets
4. **Magnitude** - Vector magnitude
5. **Radius** - Distance metric
6. **Covariance** - Joint variability
7. **Correlation Coefficient (PCC)** - Linear relationship

### Feature Naming Convention
Features follow the pattern: `{Base}_{Measure}_L0.1_{Statistic}`

Examples:
- `MI_dir_L0.1_weight` - Mutual Information direction weight
- `HH_L0.1_mean` - Host-to-Host packet mean
- `HH_jit_L0.1_variance` - Jitter variance
- `HpHp_L0.1_magnitude` - Host-to-Host payload magnitude

### Complete Feature List
1. `MI_dir_L0.1_weight`, `MI_dir_L0.1_mean`, `MI_dir_L0.1_variance`
2. `H_L0.1_weight`, `H_L0.1_mean`, `H_L0.1_variance`
3. `HH_L0.1_weight`, `HH_L0.1_mean`, `HH_L0.1_std`, `HH_L0.1_magnitude`, `HH_L0.1_radius`, `HH_L0.1_covariance`, `HH_L0.1_pcc`
4. `HH_jit_L0.1_weight`, `HH_jit_L0.1_mean`, `HH_jit_L0.1_variance`
5. `HpHp_L0.1_weight`, `HpHp_L0.1_mean`, `HpHp_L0.1_std`, `HpHp_L0.1_magnitude`, `HpHp_L0.1_radius`, `HpHp_L0.1_covariance`, `HpHp_L0.1_pcc`

## Class Labels

**Binary Classification:**
- **0** = Attack (Malicious traffic)
- **1** = Normal (Benign traffic)

### Class Distribution

#### Dataset 1 (NoDuplicates)
- **Attack samples:** 1,913,077 (78.84%)
- **Normal samples:** 513,497 (21.16%)
- **Imbalance Ratio:** ~3.73:1 (Attack:Normal)

#### Dataset 2 (Full with metadata)
- **Attack samples:** 6,506,674 (92.13%)
- **Normal samples:** 555,932 (7.87%)
- **Imbalance Ratio:** ~11.71:1 (Attack:Normal)

⚠️ **Note:** Both datasets are highly imbalanced with attacks significantly outnumbering normal traffic. Consider using techniques like:
- SMOTE (Synthetic Minority Over-sampling)
- Class weights in model training
- Stratified sampling
- Evaluation metrics: Precision, Recall, F1-Score, AUC-ROC (not just accuracy)

## Data Quality

### Missing Values
✅ **No missing values** in either dataset

### Duplicates
- **Dataset 1:** 0 duplicates (cleaned)
- **Dataset 2:** 621,659 duplicates (~8.8% of data)

### Data Types
- **Numerical Features:** All 23 engineered features are float64
- **Categorical Features (Dataset 2 only):** Device_Name, Attack, Attack_subType (object type)
- **Label:** Integer (0 or 1)

### Feature Statistics

#### Feature Value Ranges
| Feature | Min | Max | Mean |
|---------|-----|-----|------|
| MI_dir_L0.1_weight | 1.0 | 8,947.0 | 3,610.2 |
| MI_dir_L0.1_mean | 60.0 | 1,402.0 | 217.6 |
| HH_L0.1_weight | 1.0 | 7,945.0 | 1,676.1 |
| HH_jit_L0.1_mean | 0.002 | 1.53×10⁹ | 5.55×10⁸ |
| HH_jit_L0.1_variance | 0.0 | 5.88×10¹⁷ | 4.17×10¹⁵ |

#### Highly Skewed Features (|skew| > 1)
Many features exhibit high skewness, indicating non-normal distributions:
- `HpHp_L0.1_radius`: 13.93
- `HH_jit_L0.1_variance`: 11.12
- `HpHp_L0.1_covariance`: 9.64
- `HpHp_L0.1_std`: 9.59
- `HH_L0.1_radius`: 9.16

**Recommendation:** Consider log transformation or normalization for machine learning models.

## Feature Correlations

### Perfect Correlations (1.0)
Several feature pairs are perfectly correlated, indicating potential redundancy:
- `MI_dir_L0.1_weight` ↔ `H_L0.1_weight`
- `MI_dir_L0.1_mean` ↔ `H_L0.1_mean`
- `MI_dir_L0.1_variance` ↔ `H_L0.1_variance`
- `HH_L0.1_weight` ↔ `HH_jit_L0.1_weight`

### High Correlations (> 0.97)
- `HH_L0.1_mean` ↔ `HpHp_L0.1_mean`: 0.997
- `HH_L0.1_magnitude` ↔ `HpHp_L0.1_magnitude`: 0.991
- `HpHp_L0.1_mean` ↔ `HpHp_L0.1_magnitude`: 0.977

**Recommendation:** Consider feature selection or dimensionality reduction (PCA) to remove redundant features.

## Dataset Differences

### Common Features
Both datasets share **24 common columns** (all 23 engineered features + label).

### Unique to Dataset 1
- `Unnamed: 0` - Row index column

### Unique to Dataset 2
- `Device_Name` - IoT device identifier
- `Attack` - Attack family (mirai, gafgyt, Normal)
- `Attack_subType` - Specific attack variant

### Size Comparison
- Dataset 1: 2,426,574 rows (34.4% of Dataset 2)
- Dataset 2: 7,062,606 rows (complete dataset)
- Difference: 4,636,032 additional samples in Dataset 2

## Use Cases

This dataset is ideal for:

1. **Binary Classification**
   - Distinguish between attack and normal traffic
   - Train supervised learning models (Random Forest, XGBoost, Neural Networks)

2. **Multi-class Classification**
   - Identify specific attack types (Mirai vs Gafgyt)
   - Classify attack subtypes (UDP, TCP, SYN, etc.)

3. **Anomaly Detection**
   - Unsupervised learning to detect unusual patterns
   - One-class SVM, Isolation Forest

4. **Device-specific IDS**
   - Train separate models for different IoT devices
   - Transfer learning across device types

5. **Feature Importance Analysis**
   - Identify most discriminative features for botnet detection
   - Reduce feature space for real-time detection

## Machine Learning Considerations

### Preprocessing Recommendations
1. **Handle Class Imbalance**
   - Use class weights
   - Apply SMOTE or ADASYN
   - Consider ensemble methods

2. **Feature Scaling**
   - Standardization (Z-score normalization)
   - Min-Max scaling
   - Robust scaling for outliers

3. **Feature Engineering**
   - Remove perfectly correlated features
   - Apply PCA for dimensionality reduction
   - Log transform highly skewed features

4. **Train-Test Split**
   - Use stratified splitting to maintain class ratios
   - Consider temporal splitting if timestamp information is available
   - Recommended split: 70-30 or 80-20

### Evaluation Metrics
Given the class imbalance, **DO NOT rely solely on accuracy**. Use:
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under ROC curve
- **Confusion Matrix:** Detailed breakdown of predictions
- **Cohen's Kappa:** Inter-rater agreement measure

### Baseline Models
Suggested starting points:
- Logistic Regression
- Random Forest
- XGBoost/LightGBM
- Multi-layer Perceptron (MLP)
- LSTM (if temporal patterns exist)

## Technical Details

### Time Window Configuration
- **Window Duration:** 10 seconds
- **Decay Factor:** 0.1 (L0.1)
- This configuration balances between capturing enough traffic patterns and maintaining temporal relevance

### Data Format
- **File Format:** CSV (Comma-Separated Values)
- **Encoding:** UTF-8
- **Delimiter:** Comma (,)
- **Header:** First row contains column names

## Citation & References

If you use this dataset, please cite the relevant papers:

```
@article{koroniotis2019towards,
  title={Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset},
  author={Koroniotis, Nickolaos and Moustafa, Nour and Sitnikova, Elena and Turnbull, Benjamin},
  journal={Future Generation Computer Systems},
  volume={100},
  pages={779--796},
  year={2019},
  publisher={Elsevier}
}
```

### Related Papers
- [2] Detection of IoT Botnet Attacks N-BaIoT
- [3] Network Forensics for IoT
- [4] Botnet Detection in IoT Networks
- [5] Statistical Feature Engineering for IDS

## Dataset Statistics Summary

| Metric | Dataset 1 (NoDuplicates) | Dataset 2 (Full) |
|--------|-------------------------|------------------|
| Total Samples | 2,426,574 | 7,062,606 |
| Features | 23 + 1 label | 23 + 3 metadata + 1 label |
| Attack Samples | 1,913,077 (78.84%) | 6,506,674 (92.13%) |
| Normal Samples | 513,497 (21.16%) | 555,932 (7.87%) |
| Missing Values | 0 | 0 |
| Duplicates | 0 | 621,659 |
| File Size | 463 MB | 2.7 GB |
| Devices | All combined | 9 individual devices |
| Attack Types | Binary | Mirai + Gafgyt |
| Attack Subtypes | Not specified | 8 variants |

## Getting Started

### Loading the Dataset

```python
import pandas as pd

# Load Dataset 1 (No duplicates, cleaner)
df1 = pd.read_csv('BotNeTIoT-L01_label_NoDuplicates.csv')

# Load Dataset 2 (Full with metadata)
df2 = pd.read_csv('BoTNeTIoT-L01-v2.csv')

# Basic info
print(f"Dataset 1 shape: {df1.shape}")
print(f"Dataset 2 shape: {df2.shape}")

# Check class distribution
print("\nClass distribution:")
print(df1['label'].value_counts())
```

### Basic EDA

```python
import numpy as np

# Separate features and labels
X = df1.drop(['label', 'Unnamed: 0'], axis=1)
y = df1['label']

# Check for missing values
print(f"Missing values: {X.isnull().sum().sum()}")

# Basic statistics
print(X.describe())

# Check correlations
corr_matrix = X.corr()
high_corr = (corr_matrix.abs() > 0.9).sum() - 1  # subtract diagonal
print(f"\nFeatures with high correlation: {high_corr[high_corr > 0]}")
```

### Simple Classification Example

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Prepare data
X = df1.drop(['label', 'Unnamed: 0'], axis=1)
y = df1['label']

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Future Work

Potential research directions:
1. Real-time detection systems with low-latency requirements
2. Transfer learning across different IoT device types
3. Federated learning for privacy-preserving IDS
4. Deep learning architectures (CNN, LSTM, Transformers)
5. Explainable AI for interpretable threat detection
6. Integration with network security tools (Snort, Suricata)
7. Adversarial robustness against evasion attacks

## License & Usage

Please ensure you have the appropriate rights to use this dataset. Check the original dataset source for licensing information.

## Contact & Support

For questions about the dataset or collaboration opportunities, please refer to the original data source and publication.

---

**Last Updated:** February 2026  
**Version:** 1.0  
**Maintained by:** Neural Network Course Project Team
# IoT-data-EdA
