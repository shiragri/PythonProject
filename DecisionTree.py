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

# 4. מימוש Decision Tree
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.unique(y, return_counts=True)[0][0]

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.unique(y, return_counts=True)[0][0]

        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = float('inf')

        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                gini = self._calculate_gini(X[feature], y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, feature_col, y, threshold):
        left_indices = feature_col <= threshold
        right_indices = feature_col > threshold

        left_gini = 1.0 - sum((np.sum(y[left_indices] == cls) / len(y[left_indices]))**2 for cls in np.unique(y))
        right_gini = 1.0 - sum((np.sum(y[right_indices] == cls) / len(y[right_indices]))**2 for cls in np.unique(y))

        total_gini = (len(y[left_indices]) / len(y)) * left_gini + (len(y[right_indices]) / len(y)) * right_gini
        return total_gini

    def predict(self, X):
        return X.apply(lambda row: self._traverse_tree(row, self.tree), axis=1)

    def _traverse_tree(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        if row[tree['feature']] <= tree['threshold']:
            return self._traverse_tree(row, tree['left'])
        else:
            return self._traverse_tree(row, tree['right'])

# אימון המודל
dt_classifier = DecisionTreeClassifier(max_depth=5)
dt_classifier.fit(X_train, y_train)

# חיזוי על סט האימון והבדיקה
y_pred_train = dt_classifier.predict(X_train)
y_pred_test = dt_classifier.predict(X_test)

# 5. חישוב המטריקות
# פונקציה לחישוב ביצועים
def calculate_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score

# חישוב המטריקות
train_metrics = calculate_metrics(y_train.values, y_pred_train)
test_metrics = calculate_metrics(y_test.values, y_pred_test)

# הדפסת התוצאות
print("Train Metrics (Accuracy, Precision, Recall, F1-Score):", train_metrics)
print("Test Metrics (Accuracy, Precision, Recall, F1-Score):", test_metrics)
