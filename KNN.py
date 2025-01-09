
import pandas as pd
import numpy as np
import math

# 1. העלאת הנתונים
file_path = 'loan_approval_dataset.csv'  # עדכני את הנתיב

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

# 4. מימוש KNN
# פונקציית חיזוי KNN
def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []

    for test_point in X_test.values:  # עבור כל נקודה בסט הבדיקה
        distances = []

        for i, train_point in enumerate(X_train.values):  # חישוב מרחק לכל נקודה בסט האימון
            distance = math.sqrt(np.sum((train_point - test_point) ** 2))
            distances.append((distance, y_train.iloc[i]))

        # מיון המרחקים ובחירת k השכנים הקרובים ביותר
        distances = sorted(distances, key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:k]]

        # קביעת הקטגוריה לפי הצבעת רוב
        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(prediction)

    return np.array(predictions)

# 5. חישוב המטריקות
# פונקציה לחישוב ביצועים
def calculate_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score

# 6. חיפוש k הטוב ביותר
# בדיקת ביצועים על סט אימות

def find_best_k(X_train, y_train, X_val, y_val, k_values):
    best_k = None
    best_accuracy = 0
    results = {}

    for k in k_values:
        y_pred_val = knn_predict(X_train, y_train, X_val, k)
        accuracy = np.mean(y_val.values == y_pred_val)
        results[k] = accuracy

        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    return best_k, results

# בדיקת ערכי k
k_values = (1,3,5,7)
best_k, validation_results = find_best_k(X_train, y_train, X_val, y_val, k_values)

# 7. חישוב ביצועים על סט הבדיקה
# חיזוי לסט הבדיקה עם k הטוב ביותר
y_pred_test = knn_predict(X_train, y_train, X_test, k=best_k)

test_metrics = calculate_metrics(y_test.values, y_pred_test)

# הדפסת התוצאות
print("Best k:", best_k)
print("Validation Results:", validation_results)
print("Test Metrics (Accuracy, Precision, Recall, F1-Score):", test_metrics)
