# Bank Marketing Term Deposit Prediction & Customer Segmentation

[![Open In Colab](https://google.com)](https://google.com)

**Author:** Timothy Ogbudu  
**Group:** 25 (Solo)  

##  Project Overview
This project analyzes bank marketing campaign data to predict whether a client will subscribe to a term deposit ('deposit') and segments customers into actionable personas to optimize marketing spend. By combining supervised classification with unsupervised K-Means clustering, this analysis identifies both individual conversion likelihood and broader customer behavioral segments.

## 📊 Dataset
* **Source:** [Bank Marketing Dataset](https://kaggle.com) via 'kagglehub'.
* **Dimensions:** 11,162 rows and 17 features (11,092 rows after cleaning).
* **Target Variable:** 'deposit' ('yes' = 5,289 / ~47.4%, 'no' = 5,873 / ~52.6%).

## 🛠️ Methodology & Workflow

### 1. Exploratory Data Analysis & Data Cleaning
* **Leakage Prevention:** Dropped the 'duration' column. Although it showed the strongest positive correlation with 'deposit' (r = 0.4519), call duration is unknown before a call takes place and would cause severe data leakage in production.
* **Feature Engineering:** 
  * Created a binary indicator 'was_contacted_in_prev_campaign' from 'pdays' (where -1 indicated no prior contact) and replaced -1 with NaN for proper numerical scaling.
  * Relabeled `unknown` in `poutcome` as `new_customer` to preserve structural customer history.
* **Missing Values:** Removed 70 rows (~0.6% of data) where `job` was `unknown` to preserve demographic integrity without introducing imputation noise.

### 2. Customer Segmentation (K-Means Clustering)
* Scaled numerical features ('age', 'balance', 'day', 'campaign', 'previous', 'was_contacted_in_prev_campaign') using StandardScaler.
* Selected **$k = 4$ clusters** over $k = 2$ (despite a lower silhouette score of `0.2429` vs. `0.3572`) to capture granular, actionable business personas rather than a simple binary split.

### 3. Predictive Modeling & Imbalance Handling
* Built a ColumnTransformer preprocessing pipeline using SimpleImputer, StandardScaler for numerical features, and OneHotEncoder for nominal categorical features.
* Evaluated multiple classifiers against a Stratified Dummy baseline, and tested class weighting (`class_weight='balanced'`) and `SMOTE` oversampling.

---

##  Key Results & Findings

### Model Performance Comparison
The **Baseline Random Forest Classifier** outperformed all other models across Accuracy, F1-score, and ROC AUC. Because the dataset is relatively balanced (~47.3% positive class), explicit imbalance techniques (Class Weighting and SMOTE) yielded minimal changes and slightly reduced the F1-score.

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | **0.717** | **0.730** | **0.638** | **0.681** | **0.757** |
| Random Forest (SMOTE) | 0.715 | 0.733 | 0.626 | 0.675 | 0.766 |
| Random Forest (Weighted) | 0.711 | 0.729 | 0.620 | 0.670 | 0.768 |
| Logistic Regression | 0.681 | 0.699 | 0.575 | 0.631 | 0.745 |
| K-Nearest Neighbors | 0.659 | 0.659 | 0.581 | 0.618 | 0.702 |
| Dummy Classifier (Stratified) | 0.497 | 0.469 | 0.477 | 0.473 | 0.496 |

* **Top Predictive Features:** 'balance' (0.154), 'age' (0.134), 'day' (0.118), 'campaign' (0.066), and 'poutcome_success' (0.033).
* **Demographic Insight:** `retired` customers held the highest average account balance (~2,417), making them a prime demographic for deposit campaigns.

### Customer Segments Discovered ($k = 4$)
Combining K-Means clusters with target conversion rates revealed distinct customer behaviors:

| Cluster | Persona Name | Avg. Balance | Avg. Contacts (`campaign`) | Conversion Rate (`deposit = yes`) |
| :---: | :--- | :---: | :---: | :---: |
| **Cluster 3** | **Previously Contacted Engaged Customers** | 1,787 | 1.89 | **67.02%** |
| **Cluster 0** | **Lower Balance, Early Campaigners** | 1,183 | 2.07 | **44.26%** |
| **Cluster 1** | **Higher Balance, Late Initiated New Customers** | 1,701 | 2.31 | **38.83%** |
| **Cluster 2** | **High Effort New Customers** | 1,341 | 12.64 | **24.40%** |

---

##  Strategic Recommendations
1. **Deploy Baseline Random Forest:** Use the baseline Random Forest model to score leads prior to outreach.
2. **Prioritize Cluster 3:** Focus marketing resources on previously engaged customers (Cluster 3), who convert at **67.02%** with fewer than 2 contacts on average.
3. **Cap Outreach on Cluster 2:** Set contact limits for new leads requiring excessive follow-ups (Cluster 2), as they consume heavy campaign resources (~12.6 calls) while converting at only **24.40%**.

##  How to Run
1. **Run in Google Colab:** Click the **Open In Colab** badge at the top of this README (full runtime is ~1 minute).
2. **Run Locally / GitHub Codespaces:**
   ```bash
   git clone https://github.com
   cd altexamnew
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn kagglehub
   jupyter notebook marketingdataanalysis.ipynb
   ```
