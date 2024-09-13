
# Credit Card Fraud Detection Using Machine Learning Algorithms
## Project Overview: 
<p align="justify">
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
<p align="justify">
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
</p>


# Credit Card Fraud Detection Using Machine Learning Algorithms
## Descripción del Proyecto:
<p align="justify"> 
El objetivo de este proyecto es evaluar diferentes modelos de aprendizaje automático para detectar transacciones fraudulentas con tarjetas de crédito. El fraude con tarjetas de crédito es un problema significativo para las instituciones financieras y los consumidores, y la detección temprana de transacciones fraudulentas es crucial para prevenir pérdidas financieras. En este proyecto, evaluaremos tres algoritmos diferentes de aprendizaje supervisado — Random Forest, Regresión Logística, y Máquina de Vectores de Soporte (SVM) — para identificar patrones en los datos de transacciones que indiquen fraude. Sin embargo, la implementación real del modelo seleccionado se llevará a cabo en un proyecto separado.
Conjunto de datos: El conjunto de datos consiste en registros de transacciones anonimizadas, donde cada registro contiene varias características (por ejemplo, monto de la transacción, hora de la transacción, y otras características anonimizadas) junto con una etiqueta binaria que indica si la transacción fue legítima (0) o fraudulenta (1). Fue descargado de https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/data

##Explicación de las características:
Las columnas encontradas en el conjunto de datos son:

distance_from_home - la distancia desde el hogar donde ocurrió la transacción.

distance_from_last_transaction - la distancia desde la última transacción.

ratio_to_median_purchase_price - Relación del precio de la transacción al precio de compra mediano.

repeat_retailer - Si la transacción ocurrió en el mismo minorista.

used_chip - Si la transacción se realizó a través de un chip (tarjeta de crédito).

used_pin_number - Si la transacción se realizó utilizando un número PIN.

online_order - Si la transacción es un pedido en línea.

fraud - Si la transacción es fraudulenta.

## Enfoque:
#### Preprocesamiento de Datos:
Limpiar y preprocesar los datos, manejando los valores faltantes y escalando las características numéricas para un mejor rendimiento del modelo. (En este caso, los datos ya estaban limpios y completos).
EDA: En este paso, encontramos que el fraude estaba fuertemente correlacionado con la 'Relación con el precio de compra mediano'; además:
Transacciones totales: 1,000,000
Total de casos de fraude: 87,403.0
Total de fraudes con chip: 22,410.0 de 1,000,000
Total de fraudes con PIN: 273.0 de 1,000,000
Tasa de fraude para transacciones en línea: 12.71% (82,711 casos de 650,552.0 transacciones en línea)
Tasa de fraude para transacciones fuera de línea: 1.34% (4,692 casos de 349,448.0 transacciones fuera de línea)
Dividir el conjunto de datos en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo.

## Entrenamiento del Modelo:
### Regresión Logística:
Utilizado como modelo base para comprender la relación entre las características y la probabilidad de una transacción fraudulenta.

### Random Forest:
Un potente método de aprendizaje en conjunto que combina múltiples árboles de decisión para capturar patrones complejos en los datos.

### Support Vector Machine (SVM):
Un robusto algoritmo de clasificación que encuentra un hiperplano óptimo para separar transacciones fraudulentas y legítimas. En este caso particular, utilizamos una muestra de los datos debido al tiempo necesario para procesarlos.

## Evaluación del Modelo:
<p align="justify"> 
   En este paso evaluamos el rendimiento de cada modelo utilizando métricas como precisión, exhaustividad, recall y F1-score. El objetivo es equilibrar estas métricas para minimizar los falsos positivos (marcar incorrectamente una transacción legítima como fraude) y los falsos negativos (no detectar un fraude real). Obtuvimos mejores resultados con el algoritmo Máquina de Vectores de Soporte (SVM), considerando que fue entrenado con un subconjunto de los datos. Como se mencionó anteriormente, la implementación se llevará a cabo en un proyecto posterior.
### Regresión Logística - Métricas de Evaluación
Precisión: 0.8902
Recall: 0.5974
F1 Score: 0.7150
Exactitud: 0.9585
### Máquina de Soporte Vectorial - Métricas de Evaluación
Precisión: 0.8978
Recall: 0.5834
F1 Score: 0.7073
Exactitud: 0.9579
</p>
