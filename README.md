# Credit Card Fraud Detection Using Machine Learning Algorithms
## Project Overview: 
The goal of this project is to develop a machine learning model to detect fraudulent credit card transactions. Credit card fraud is a significant problem for financial institutions and consumers, and early detection of fraudulent transactions is crucial to prevent financial losses. In this project, we will use three different machine learning algorithms — Random Forest, Logistic Regression, and Support Vector Machine (SVM) — to identify patterns in transaction data that indicate fraud.

Dataset: The dataset consists of anonymized transaction records, with each record containing various features (e.g., transaction amount, time of transaction, and other anonymized features) along with a binary label indicating whether the transaction was legitimate (0) or fraudulent (1). It was downloaded from https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/data

### Feature Explanation:

distance_from_home - the distance from home where the transaction happened.

distance_from_last_transaction - the distance from last transaction happened.

ratio_to_median_purchase_price - Ratio of purchased price transaction to median purchase price.

repeat_retailer - Is the transaction happened from same retailer.

used_chip - Is the transaction through chip (credit card).

used_pin_number - Is the transaction happened by using PIN number.

online_order - Is the transaction an online order.

fraud - Is the transaction fraudulent.

## Approach:

### Data Preprocessing:

- Clean and preprocess the data, handling missing values and scaling numerical features for better model performance. (In this scenario, data was already cleaned and complete)
- EDA: In this step we found 
- Split the dataset into training and testing sets to evaluate model performance.
  
### Model Training:

#### Logistic Regression:
Used as a baseline model to understand the relationship between features and the likelihood of a fraudulent transaction.
#### Random Forest: 
A powerful ensemble learning method that combines multiple decision trees to capture complex patterns in the data.
#### Support Vector Machine (SVM): 
A robust classification algorithm that finds an optimal hyperplane to separate fraudulent and legitimate transactions.
Model Evaluation:

Evaluate each model's performance using metrics such as accuracy, precision, recall, and F1-score. The goal is to balance these metrics to minimize false positives (incorrectly flagging a legitimate transaction as fraud) and false negatives (failing to detect actual fraud).
