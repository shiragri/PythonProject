import pandas as pd
import numpy as np
import math

# 1. העלאת הנתונים
file_path = 'loan_approval_dataset.csv'

data = pd.read_csv(file_path)

# 2. ניקוי ונירמול הנתונים
# פונקציה לנירמול עמודות
def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min())

# נירמול כל העמודות למעט עמודת המטרה
for col in data.columns[:-1]:
    data[col] = normalize_column(data[col])

# 3. חלוקת הנתונים ל-Train, Validation, ו-Test
# פונקציה לחלוקה ידנית
def train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_size = int(len(X) * test_size)
    val_size = int(len(X) * val_size)

    train_indices = indices[:-(test_size + val_size)]
    val_indices = indices[-(test_size + val_size):-test_size]
    test_indices = indices[-test_size:]

    X_train, X_val, X_test = X.iloc[train_indices], X.iloc[val_indices], X.iloc[test_indices]
    y_train, y_val, y_test = y.iloc[train_indices], y.iloc[val_indices], y.iloc[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test

# חלוקה
X = data.drop(columns=[' loan_status'])  # מאפיינים
y = data[' loan_status']  # משתנה מטרה

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, test_size=0.2, val_size=0.1)

# 4. מימוש Naive Bayes
# חישוב הסתברויות מותנות
class NaiveBayesClassifier:
    def __init__(self):
        self.prior_probs = {}
        self.conditional_probs = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior_probs = {cls: np.mean(y == cls) for cls in self.classes}

        self.conditional_probs = {}
        for cls in self.classes:
            class_data = X[y == cls]
            self.conditional_probs[cls] = {
                col: self._calculate_conditional_probs(class_data[col]) for col in X.columns
            }

    def _calculate_conditional_probs(self, col):
        unique_values = np.unique(col)
        probs = {val: np.mean(col == val) for val in unique_values}
        return probs

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            class_probs = {}
            for cls in self.classes:
                prob = self.prior_probs[cls]
                for col in X.columns:
                    value = row[col]
                    prob *= self.conditional_probs[cls][col].get(value, 1e-6)  # לטפל בערכים נדירים
                class_probs[cls] = prob
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

# אימון המודל
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# חיזוי על סט האימון והבדיקה
y_pred_train = nb_classifier.predict(X_train)
y_pred_test = nb_classifier.predict(X_test)

# 5. חישוב המטריקות
# פונקציה לחישוב ביצועים
def calculate_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return float( round(accuracy,3) ),float( round(precision,3)) ,float( round(recall,3 )) ,float( round(f1_score,3))

# חישוב המטריקות
train_metrics = calculate_metrics(y_train.values, y_pred_train)
test_metrics = calculate_metrics(y_test.values, y_pred_test)

# הדפסת התוצאות
print("Train Metrics- Accuracy:"+str( train_metrics[0])+",  Precision:"+str( train_metrics[1])+",  Recall:"+str (train_metrics[2])+",  F1-Score:"+str( train_metrics[3]))
print("Test Metrics- Accuracy:"+str( test_metrics[0])+",  Precision:"+str( test_metrics[1])+",  Recall:"+str (test_metrics[2])+",  F1-Score:"+str( test_metrics[3]))


