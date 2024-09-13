# Credit Card Fraud Detection Using Machine Learning Algorithms
## Project Overview: 
The goal of this project is to evaluate different machine learning models to detect fraudulent credit card transactions. Credit card fraud is a significant problem for financial institutions and consumers, and early detection of fraudulent transactions is crucial to prevent financial losses. In this project, we will evaluate three different supervised machine learning algorithms — Random Forest, Logistic Regression, and Support Vector Machine (SVM) — to identify patterns in transaction data that indicate fraud. However, the actual implementation of the selected model will be undertaken in a separate project.

Dataset: The dataset consists of anonymized transaction records, with each record containing various features (e.g., transaction amount, time of transaction, and other anonymized features) along with a binary label indicating whether the transaction was legitimate (0) or fraudulent (1). It was downloaded from https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/data

### Feature Explanation:
The columns found on the data set are:

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
- EDA: In this step we found that Fraud was strongly correlated to 'Ratio to median purchase price' also:
   - Total Transactions 1000000
   - total fraud cases 87403.0
   - total chip frauds 22410.0 out of 1000000
   - total pin frauds 273.0 out of 1000000
   - Fraud rate for online transactions: 12.71% (82711 cases out of 650552.0 online  transactions)
   - Fraud rate for offline transactoins: 1.34% (4692 cases out of 349448.0 offline  transactions)
- Split the dataset into training and testing sets to evaluate model performance.
  
### Model Training:

#### Logistic Regression:
Used as a baseline model to understand the relationship between features and the likelihood of a fraudulent transaction.
#### Random Forest: 
A powerful ensemble learning method that combines multiple decision trees to capture complex patterns in the data.
#### Support Vector Machine (SVM): 
A robust classification algorithm that finds an optimal hyperplane to separate fraudulent and legitimate transactions. In this particular case we used a sample of the data because of the time to process.


### Model Evaluation:
In this step we evaluate each model's performance using metrics such as accuracy, precision, recall, and F1-score. The goal is to balance these metrics to minimize false positives (incorrectly flagging a legitimate transaction as fraud) and false negatives (failing to detect actual fraud). We got better results with Support Vector Machine (SVM) considering that it was trained with a subset of the data. As it was mentioned previosly the implementation will be undertaken in a subsequent project.

#### Logistic Regression - Evaluation Metrics
- Precision: 0.8902
- Recall: 0.5974
- F1 Score: 0.7150
- Accuracy: 0.9585

#### Support Vector Machine - Evaluation Metrics
- Precision: 0.8978
- Recall: 0.5834
- F1 Score: 0.7073
- Accuracy: 0.9579
