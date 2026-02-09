# 💳 Credit Card Risk — Analysis and Prediction

## 📌 Context

This project uses the **Credit Card Approval** dataset available on Kaggle:

🔗 Dataset:  
https://www.kaggle.com/datasets/youssefelbadry10/credit-card-approval/data

The objective is to perform an **Exploratory Data Analysis (EDA)** to understand the data and then build **machine learning models** to predict whether a customer is classified as **high risk**.

---

# 📊 Exploratory Data Analysis (EDA)

## 📌 Dataset Overview

The dataset contains **29165 observations** and **20 features**, combining demographic, socioeconomic, behavioral, and account-related information. The target variable is **`Is high risk`**, a binary indicator used for credit risk prediction.

### Feature Groups
- **Identifier**: `ID`
- **Demographic**: Gender, Age, Marital status, Family member count, Children Count
- **Socioeconomic**: Income, Employment status, Employment length, Education level, Job title
- **Assets & Accessibility**: Has a car, Has a property, phone/email indicators
- **Housing**: Dwelling
- **Behavioral / Relationship**: Account age
- **Target**: Is high risk

---

All the EDA procedure can be seen in the Jupyther notebook attached.

## ✅ EDA Conclusions

- The target variable is highly imbalanced, only 1.7% of the total data is flagged as a high-risk, while 98,3% of the total data is not a high-risk situation. This represents a overall big challenge to ML models since there is a strong motivation to keep all the classifications as a ''not a high risk'' prediction;
- No single feature strongly predicts risk in isolation, such that all the predictive power is expected to comes from **feature interactions**.


### EDA-driven Cleaning Actions
- Some features as `ID` and `Children Count` where dropped from the dataframe, since the first is not relevant and the second is high correlated with `Family member count`.
- The `Age`, `Employment length` and `Account age` needed to be corrected and carefully studied in order to identify some outliers that represented wrong data.
- Some minor changes on `Has a Car` and `Has a property`. 

---

## 🚀 Machine Learning Approach to Predict Credit Risk

After all the cleaning, some machine learning methods from `scikit-learn` were employed to try to predict a real scenario:
- LogisticRegression
- RandomForestClassifier
- LinearSVC
- DecisionTreeClassifier
- GradientBoostingClassifier
- KNeighborsClassifier

