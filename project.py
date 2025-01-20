import pandas as pd
import numpy as np
from NaiveBayes import NaiveBayesClassifier
from DecisionTree import DecisionTreeClassifier
from KNN import KNNClassifier



# Uploading the data.
file_path = 'loan_approval_dataset.csv'

data = pd.read_csv(file_path)


# Function for normalizing columns.
def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min())


#Normalize all columns except the prediction column.
for col in data.columns[:-1]:
    data[col] = normalize_column(data[col])

#Splitting into data columns and prediction column.
x = data.drop(columns=[' loan_status'])
y = data[' loan_status']



#  Splitting the data into Train, Validation, and Test.
def train_val_test_split(x, y, test_size=0.2, val_size=0.2, random_state=5):
    np.random.seed(random_state)
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    test_size = int(len(x) * test_size)
    val_size = int(len(x) * val_size)

    train_indices = indices[:-(test_size + val_size)]
    val_indices = indices[-(test_size + val_size):-test_size]
    test_indices = indices[-test_size:]


    x_train = x.iloc[train_indices]
    x_val = x.iloc[val_indices]
    x_test = x.iloc[test_indices]
    y_train =y.iloc[train_indices]
    y_val =y.iloc[val_indices]
    y_test = y.iloc[test_indices]

    return x_train, x_val, x_test, y_train, y_val, y_test



# Function to calculate success percentages.
def calculate_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return float( round(accuracy,3) ),float( round(precision,3)) ,float( round(recall,3 )) ,float( round(f1_score,3))


###############################################
#               Naive Bayes
###############################################

x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y, test_size=0.2, val_size=0)

# Training the model.
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(x_train, y_train)

# Prediction on the training and testing sets.
y_pred_train = nb_classifier.predict(x_train)
y_pred_test = nb_classifier.predict(x_test)

# Calculating success percentages.
train_metrics = calculate_metrics(y_train.values, y_pred_train)
test_metrics = calculate_metrics(y_test.values, y_pred_test)

# Printing the results for Naive Bayes.
print("Naive Bayess results:")
print("Train Metrics- Accuracy:"+str( train_metrics[0])+",  Precision:"+str( train_metrics[1])+",  Recall:"+str (train_metrics[2])+",  F1-Score:"+str( train_metrics[3]))
print("Test Metrics- Accuracy:"+str( test_metrics[0])+",  Precision:"+str( test_metrics[1])+",  Recall:"+str (test_metrics[2])+",  F1-Score:"+str( test_metrics[3]))

###############################################
#               Decision Tree
###############################################

x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y, test_size=0.2, val_size=0)

# Training the model.
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train, y_train)

#Prediction on the training and testing sets.
dt_y_pred_train = dt_classifier.predict(x_train)
dt_y_pred_test = dt_classifier.predict(x_test)

# Calculating success percentages.
dt_train_metrics = calculate_metrics(y_train.values, dt_y_pred_train)
dt_test_metrics = calculate_metrics(y_test.values, dt_y_pred_test)

#Printing the results for Decision Tree.
print("")
print("Decision Tree results:")
print("Train Metrics- Accuracy:"+str( dt_train_metrics[0])+",  Precision:"+str( dt_train_metrics[1])+",  Recall:"+str (dt_train_metrics[2])+",  F1-Score:"+str( dt_train_metrics[3]))
print("Test Metrics- Accuracy:"+str( dt_test_metrics[0])+",  Precision:"+str( dt_test_metrics[1])+",  Recall:"+str (dt_test_metrics[2])+",  F1-Score:"+str( dt_test_metrics[3]))

###############################################
#               KNN
###############################################

x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y, test_size=0.2, val_size=0.2)

# Evaluating performance for different values of K.
k_values = [1, 3, 5, 7]
validation_results = {}

for k in k_values:
    knn_classifier = KNNClassifier(k=k)
    knn_classifier.fit(x_train, y_train)
    y_pred_val = knn_classifier.predict(x_val,k)
    validation_metrics = calculate_metrics(y_val.values, y_pred_val)
    validation_results[k] = validation_metrics

# Finding the K with the best performance.
best_k = max(validation_results, key=lambda k: validation_results[k][0])

#Training the model with the best K.
knn_classifier = KNNClassifier(k=best_k)
knn_classifier.fit(x_train, y_train)

# Prediction on the training and testing sets.
knn_y_pred_train = knn_classifier.predict(x_train)
knn_y_pred_test = knn_classifier.predict(x_test)

# Calculating success percentages.
knn_train_metrics = calculate_metrics(y_train.values, knn_y_pred_train)
knn_test_metrics = calculate_metrics(y_test.values, knn_y_pred_test)

# Printing the results for KNN
print("")
print("KNN results:")
print(f"Best K: {best_k}")
print("Train Metrics- Accuracy:"+str( knn_train_metrics[0])+",  Precision:"+str( knn_train_metrics[1])+",  Recall:"+str (knn_train_metrics[2])+",  F1-Score:"+str( knn_train_metrics[3]))
print("Test Metrics- Accuracy:"+str( knn_test_metrics[0])+",  Precision:"+str( knn_test_metrics[1])+",  Recall:"+str (knn_test_metrics[2])+",  F1-Score:"+str( knn_test_metrics[3]))

