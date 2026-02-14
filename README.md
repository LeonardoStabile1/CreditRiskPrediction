# [En-US] Credit Card Risk — Analysis and Prediction

##  Context

This project uses the **Credit Card Approval** dataset available on Kaggle:

Dataset:  
https://www.kaggle.com/datasets/youssefelbadry10/credit-card-approval/data

The objective is to perform an **Exploratory Data Analysis (EDA)** to understand the data and then build **machine learning models** to predict whether a customer is classified as **high risk**.

---

# Exploratory Data Analysis (EDA)

## Dataset Overview

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

## EDA Conclusions

For a detailed breakdown of the data exploration phase and key insights, please refer to the [EDA file](EDA.md).

- The target variable is highly imbalanced, only 1.7% of the total data is flagged as a high-risk, while 98,3% of the total data is not a high-risk situation. This represents a overall big challenge to ML models since there is a strong motivation to keep all the classifications as a ''not a high risk'' prediction;
- No single feature strongly predicts risk in isolation, such that all the predictive power is expected to comes from **feature interactions**.


### EDA-driven Cleaning Actions
- Some features as `ID` and `Children Count` where dropped from the dataframe, since the first is not relevant and the second is high correlated with `Family member count`.
- The `Age`, `Employment length` and `Account age` needed to be corrected and carefully studied in order to identify some outliers that represented wrong data.
- Some minor changes on `Has a Car` and `Has a property`.

All the procedure can be seen in the [Jupyter notebook](Credit%20Card%20Risk.ipynb) attached.

---

#  Supervised Machine Learning Approach to Predict Credit Risk

After all the cleaning, some machine learning methods from `scikit-learn` were employed to try to predict a real scenario:
- LogisticRegression
- RandomForestClassifier
- LinearSVC
- DecisionTreeClassifier
- GradientBoostingClassifier
- KNeighborsClassifier

The data was separated into 80% used as a train sample, while 20% used as a test of the training. The confusion matrices of each method can be observed on the Jupyther notebook. Here, I will present the general results of each model so that we can observe their general results. Since the data is highly imbalanced, the accuracy of the test is not a good measure, because a model that predicts `0` for every configuration will have a 98,3% accuracy. We actually need to observe each metrics on the following table

### High-Risk = 0 (Not a high risk)
| Model                          | Precision | Recall   | F1-Score | Support |
|--------------------------------|-----------|---------|----------|---------|
| LogisticRegression             | 0.9890    | 0.6419  | 0.7785   | 4772    |
| RandomForestClassifier         | 0.9890    | 0.9407  | 0.9642   | 4772    |
| DecisionTreeClassifier         | 0.9882    | 0.8963  | 0.9400   | 4772    |
| GradientBoostingClassifier     | 0.9847    | 0.9992  | 0.9919   | 4772    |
| KNeighborsClassifier           | 0.9861    | 0.9969  | 0.9915   | 4772    |

As we can see, all the models presents a good precision, with very few (1% ~ 1.5%) of false alarms for the results with no high risk. Also, the Recall column shows us that out of 4772 results with 0 high risk, the LogisticRegression maganed to find 64%, which is a very small result when compared to the other models. The F1-Score shows us a average of the Precision and Recall, showing that the LogisticRegression is not a good model for this result.

### High-Risk = 1 (A high risk)

| Model                          | Precision | Recall  | F1-Score | Support |
|--------------------------------|-----------|--------|----------|---------|
| LogisticRegression             | 0.0262    | 0.5750 | 0.0501   | 80      |
| RandomForestClassifier         | 0.0958    | 0.3750 | 0.1527   | 80      |
| DecisionTreeClassifier         | 0.0553    | 0.3625 | 0.0960   | 80      |
| GradientBoostingClassifier     | 0.6000    | 0.0750 | 0.1333   | 80      |
| KNeighborsClassifier           | 0.4643    | 0.1625 | 0.2407   | 80      |

The High-Risk = 1 represents only 1,7% of the total data so that their prediction is very hard to manage. We can see that the GradientBoostingClassifier and KNeighborsClassifier have a good precision, but their Recall value is very low when compared to the other models. The LogisticRegression managed to find 57,5% of the total high risk data, followed by the RandomForestClassifier, which shows a balanced result from the F1-Score.


### Overall Performance

| Model                          | Accuracy  | Weighted Avg Precision | Weighted Avg F1 | Weighted Avg Recall | Total Support |
|--------------------------------|-----------|-----------------------|----------------|-------------------|---------------|
| LogisticRegression             | 0.6408    | 0.9731                | 0.7665         | 0.6408            | 4852          |
| RandomForestClassifier         | 0.9314    | 0.9743                | 0.9509         | 0.9314            | 4852          |
| DecisionTreeClassifier         | 0.8875    | 0.9728                | 0.9261         | 0.8875            | 4852          |
| GradientBoostingClassifier     | 0.9839    | 0.9784                | 0.9777         | 0.9839            | 4852          |
| KNeighborsClassifier           | 0.9831    | 0.9775                | 0.9791         | 0.9831            | 4852          |


- LogisticRegression achieves decent precision on majority classes but struggles with recall, meaning it misses a significant portion of minority class samples. This is reflected in its low F1-score. 
- DecisionTreeClassifier improve overall accuracy and F1 compared to Logistic Regression, but their recall is still moderate. They better capture minority classes than Logistic Regression but may overfit to majority classes. 
- RandomForestClassifier shows strong performance across all weighted metrics, balancing precision and recall. It handles class imbalance better than single trees due to ensemble averaging. 
- GradientBoostingClassifier achieves the highest overall accuracy and weighted F1, excelling at predicting majority class samples. However, macro-average recall is lower, indicating some minority classes may still be underpredicted. 
- KNeighborsClassifier  also achieves excellent overall performance and slightly higher weighted F1 than Gradient Boosting. Its performance is stable across classes but may be sensitive to the distribution of the minority class in high-dimensional spaces.

# Conclusion

Given the strong class imbalance in the data, model evaluation must prioritize precision–recall trade-offs rather than accuracy alone. The results show that models achieving very high overall accuracy do so mainly by correctly classifying the majority class, while still struggling to consistently identify high-risk cases. No single model provides an optimal solution for this problem. Some approaches achieve higher recall for high-risk observations at the cost of precision, while others are highly precise but fail to capture a meaningful portion of the minority class. This highlights an inherent limitation of using a single classifier under severe imbalance. Therefore, a combined analysis using the best-performing models is preferred over relying on a single choice. This strategy allows cross-validation of predictions, reduces the risk of missing high-risk cases, and provides a more robust and reliable decision framework for real-world deployment, where false negatives in high-risk detection can have significant consequences.
