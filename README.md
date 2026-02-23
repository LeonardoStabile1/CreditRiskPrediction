# Credit Risk: Analysis and Predictions
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Lib-Pandas-150458)
![Scikit Learn](https://img.shields.io/badge/Lib-Scikit_Learn-150458)
![Status](https://img.shields.io/badge/Status-Finished-success)

This project is the English version of the [Análise e Predição de Risco de Crédito](https://github.com/LeonardoStabile1/Projeto-Analise-e-Predicao-de-Risco-de-Credito), available in Portuguese.

# Section 1:  What is the problem and why is it important? 

In consumer and commercial lending, financial institutions must decide which applicants are likely to repay their obligations and which pose a higher probability of default. Inaccurate risk assessment can lead to increased credit losses, capital misallocation, and regulatory exposure. Therefore, developing a reliable credit risk prediction model becomes essential to support underwriting decisions with data-driven, objective criteria.

This project addresses that challenge by leveraging the [Credit Card Approval](https://www.kaggle.com/datasets/youssefelbadry10/credit-card-approval/data) dataset to identify key risk drivers, distinguish low-risk applicants, and estimate default probability by means of supervised machine learning algorithms. The objective is to support more consistent approval decisions, enhance portfolio quality, and enable risk-adjusted pricing grounded in quantitative evidence.


## Main questions: 

Before proceeding, we define two central analytical questions:

### 1) What are the primary risk drivers, and which features are associated with lower risk?


To address this, we examine the categorical variables and estimate the observed risk within each category. Based on these conditional risk estimates, we compute two standard epidemiological and risk metrics: `Risk Difference` (RD) and `Relative Risk` (RR), also referred to as `Lift`.

Risk Difference measures the absolute deviation from the baseline risk, while Relative Risk measures proportional deviation. Formally, if $P(Y=y)$ is the probability of an unexposed group to have the answer $y$, and $P(Y=y|X=x)$ is the probability of the answer $y$ under the exposition of $X=x$, then

RD = P(Y = y | X = x) − P(Y = y)  
RR = P(Y = y | X = x) / P(Y = y)

These metrics allow us to identify categories that significantly increase (RR > 1) or decrease (RR < 1) default risk, both in absolute and relative terms.

### 2) How should the predictive model be evaluated and calibrated?

Because credit risk datasets are typically imbalanced, overall accuracy is not an adequate performance metric. Instead, we prioritize Recall and F1-score, which better capture performance on the minority (high-risk) class.

From a decision-theoretic perspective, model calibration must reflect the trade-off between false positives (rejecting good clients) and false negatives (approving high-risk clients). Therefore, threshold selection should align with the institution’s risk appetite and cost structure.


# Section 2: Exploratory data analysis (EDA)

## General Features  

The dataset contains **36457 observations** and **20 variables**, integrating demographic, socioeconomic, behavioral, and account-related information. The target variable, **_Is high risk_**, is a binary indicator used for credit risk prediction.

### Variable Groups  

- **Identifier:**  `ID`

- **Demographic:**  `Gender`, `Age`, `Marital status`, `Family member count`, `Children count`  

- **Socioeconomic:** `Income`, `Employment status`, `Employment length`, `Education leve`, `Job title`  

- **Assets & Accessibility:** `Has a car`, `Has a property`, `Phone/email indicators`  

- **Housing:**  `Dwelling type` 

- **Behavioral / Relationship:** `Account age`  

- **Target:**  `Is high risk` 

For a detailed description of the exploratory data analysis (EDA) process and key insights, refer to the Jupyter Notebook [Credit Card Risk.ipynb](Credit%20Card%20Risk.ipynb).

### Data Cleaning Actions Guided by EDA

- The variables `ID` and `Children Count` were removed from the dataset. The former has no analytical relevance, and the latter exhibits high correlation with `Family member count`, adding limited incremental information.

- The variables `Age`, `Employment length`, and `Account age` required correction and careful outlier analysis due to inconsistencies identified during exploration.

- Minor standardization procedures were applied to the variables `Has a Car` and `Has a property` to ensure categorical consistency.

### Exploratory Analysis Conclusions

- The target variable is highly imbalanced: only **1.7%** of the observations are classified as high risk, while **98.3%** are non–high risk. This represents a significant challenge for Machine Learning models, as there is a strong tendency to classify all observations as “non–high risk” to artificially maximize overall accuracy.

- Table 1 presents the categories most strongly associated with elevated risk, while Table 2 highlights those associated with lower relative risk levels. We observe that occupational variables appear prominently among the top risk drivers, suggesting that employment segmentation plays a central role in risk differentiation. Certain housing conditions and marital statuses also exhibit materially higher relative risk (RR > 1), indicating structural socioeconomic effects.

### Table 1: Primary Risk Drivers

| Position | Feature        | Category            | RR (%) |
|----------|---------------|---------------------|--------|
| 1        | Job title     | IT staff            | 296    |
| 2        | Job title     | Low-skill Laborers  | 270    | 
| 3        | Dwelling      | Office apartment    | 203    |
| 4        | Marital status| Widow               | 173    |
| 5        | Dwelling      | Municipal apartment | 158    |

- Conversely, the lowest-risk categories are concentrated among specific employment statuses, education levels, and professional occupations. In particular, higher education attainment and certain specialized professions are associated with substantially lower observed default incidence (RR ≈ 0), reinforcing the importance of human capital and labor stability as protective factors.


### Table 2: Features Associated with Lower Risk Levels

| Position | Feature            | Category                | RR (%) |
|----------|-------------------|--------------------------|--------|
| 51       | Employment status | Student                  | 0      |
| 50       | Education level   | Academic degree          | 0      | 
| 49       | Job title         | Realty agents            | 0      |
| 48       | Job title         | Private service staff    | 0.34   |
| 47       | Job title         | Medicine staff           | 0.48   |


# Section 3: Risk prediction using Supervised Machine Learning

After completing all data cleaning procedures, several machine learning algorithms from `scikit-learn` were implemented to simulate a real-world risk prediction scenario:

- LogisticRegression  
- RandomForestClassifier  
- LinearSVC  
- DecisionTreeClassifier  
- GradientBoostingClassifier  
- KNeighborsClassifier  

The dataset was split into **80% training** and **20% testing** sets. The confusion matrices for each model are available in the Jupyter Notebook. Below, the overall performance metrics are presented for comparative analysis.

Given the strong class imbalance, accuracy alone is not an appropriate evaluation metric. A naive model that predicts `0` (non–high risk) for all observations would already achieve 98.3% accuracy. Therefore, a detailed assessment using more informative performance metrics is required, as shown in the following tables.


# Results

## High-Risk = 0 (Non–High Risk)

| Model                      | Precision | Recall   | F1-Score |
|----------------------------|-----------|----------|----------|
| LogisticRegression         | 0.9890    | 0.6606   | 0.7924   |
| RandomForestClassifier     | 0.9904    | 0.9662   | 0.9782   |
| DecisionTreeClassifier     | 0.9877    | 0.8843   | 0.9331   |
| GradientBoostingClassifier | 0.9845    | 0.9992   | 0.9917   |
| KNeighborsClassifier       | 0.9857    | 0.9942   | 0.9899   |

All models show very high precision (≥ 0.984), meaning false positives remain close to 1–1.5%.  

However, LogisticRegression identifies only 66.06% of the non–high-risk observations, substantially underperforming compared to the other algorithms. GradientBoostingClassifier (99.92%) and KNeighborsClassifier (99.42%) nearly perfectly recover the majority class.  

The F1-score confirms this pattern: LogisticRegression (0.7924) is clearly inferior for this class, while GradientBoostingClassifier (0.9917) and KNeighborsClassifier (0.9899) achieve near-perfect balance between precision and recall.

---

## High-Risk = 1 (High Risk)

The High-Risk = 1 class represents only 1.7% of the dataset, making prediction significantly more challenging.

| Model                      | Precision | Recall  | F1-Score |
|----------------------------|-----------|---------|----------|
| LogisticRegression         | 0.0277    | 0.5876  | 0.0530   |
| RandomForestClassifier     | 0.1743    | 0.4330  | 0.2485   |
| DecisionTreeClassifier     | 0.0448    | 0.3299  | 0.0790   |
| GradientBoostingClassifier | 0.4444    | 0.0412  | 0.0755   |
| KNeighborsClassifier       | 0.2609    | 0.1237  | 0.1678   |

Model behavior differs substantially for the minority class:

- LogisticRegression achieves the highest recall (58.76%), identifying more high-risk cases, but with extremely low precision.
- RandomForestClassifier provides the best balance (highest F1-score = 0.2485).
- GradientBoostingClassifier and KNeighborsClassifier show higher precision but very low recall.
- DecisionTreeClassifier presents weak performance overall.

Overall, RandomForestClassifier offers the most balanced trade-off for identifying high-risk clients.

---

## Overall Performance

| Model                      | Accuracy | Weighted Avg Precision | Weighted Avg F1 | Weighted Avg Recall |
|----------------------------|----------|------------------------|-----------------|---------------------|
| LogisticRegression         | 0.6594   | 0.9742                 | 0.7804          | 0.6594              |
| RandomForestClassifier     | 0.9576   | 0.9772                 | 0.9663          | 0.9576              |
| DecisionTreeClassifier     | 0.8753   | 0.9724                 | 0.9193          | 0.8753              |
| GradientBoostingClassifier | 0.9836   | 0.9757                 | 0.9769          | 0.9836              |
| KNeighborsClassifier       | 0.9801   | 0.9739                 | 0.9766          | 0.9801              |

- LogisticRegression shows weak overall performance due to poor recall in the majority class.
- DecisionTreeClassifier improves performance but remains limited in minority detection.
- RandomForestClassifier demonstrates consistent and balanced performance.
- GradientBoostingClassifier achieves the highest overall accuracy, driven mainly by strong majority-class prediction.
- KNeighborsClassifier delivers strong global metrics and slightly higher weighted F1 than Gradient Boosting.

---

# Conclusion

Due to the severe class imbalance, model evaluation must prioritize the precision–recall trade-off rather than accuracy alone. High overall accuracy is largely driven by correct classification of the dominant class and does not guarantee effective detection of high-risk clients.

Among the evaluated models, RandomForestClassifier provides the best balance for the minority class, achieving the highest F1-score while maintaining reasonable recall. Nevertheless, no single model offers a fully satisfactory solution under extreme imbalance.

A more robust real-world credit risk strategy would involve threshold adjustment, cost-sensitive learning, or ensemble decision frameworks to reduce false negatives and better align model outputs with financial risk exposure.
