

import numpy as np
import pandas as pd

class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self, x, y):
        self.tree = self._build_tree(x, y, depth=0)

    def _build_tree(self, x, y, depth):
        # If all the examples belong to the same class.
        if len(np.unique(y)) == 1:
            return np.unique(y, return_counts=True)[0][0]

        # If there are no more features to split on
        if x.empty :
            return np.unique(y, return_counts=True)[0][0]

        #  search for the best feature to split on
        best_feature = self._find_best_split(x, y)
        if best_feature is None:
            return np.unique(y, return_counts=True)[0][0]

        # Finding the median of the values for the feature
        median = x[best_feature].median()

        # Splitting the data into two groups:
        # one with values less than or equal to the median, and one with values greater than the median
        left_indices = x[best_feature] <= median
        right_indices = x[best_feature] > median

        # Ensuring that the split does not result in empty groups
        if left_indices.sum() == 0 or right_indices.sum() == 0:
            return np.unique(y, return_counts=True)[0][0]

        #  Building the subtrees recursively
        left_subtree = self._build_tree(x[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(x[right_indices], y[right_indices], depth + 1)

        return {
            'feature': best_feature,
            'threshold': median,
            'left': left_subtree,
            'right': right_subtree
        }

    #Iterates over all the attributes (columns) in the data
    # calculates the Information Gain for each possible value of the attribute.
    # The attribute with the highest Information Gain will be chosen for the split
    def _find_best_split(self, x, y):
        best_feature = None
        best_threshold = None
        best_info_gain = -float('inf')

        for feature in x.columns:
            thresholds = x[feature].unique()
            for threshold in thresholds:
                info_gain = self._calculate_information_gain(x[feature], y, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature

    def _calculate_information_gain(self, feature_col, y, threshold):
        # Splitting the groups according to the threshold.
        left_indices = feature_col <= threshold
        right_indices = feature_col > threshold

        # Calculating the entropy of each group.
        def entropy(y):
            class_counts = np.bincount(y)

            # Calculating the probabilities of each class.
            total_count = len(y)
            k = len(np.unique(y))  # Number of distinct classes.
            probabilities = (class_counts + 1) / (total_count + 1 * k) # Calculating the probabilities with Laplace Smoothing.
            entropy_value = -np.sum(probabilities * np.log2(probabilities))  # Calculating the entropy

            return entropy_value
        left_entropy = entropy(y[left_indices])
        right_entropy = entropy(y[right_indices])

        # Calculating the total entropy
        total_entropy = entropy(y)
        weighted_entropy = (len(y[left_indices]) / len(y)) * left_entropy + (len(y[right_indices]) / len(y)) * right_entropy

        return total_entropy - weighted_entropy   # Information Gain


    def predict(self, x):
        predictions = []
        for _, row in x.iterrows():
            predictions.append(self._traverse_tree(row, self.tree))
        return np.array(predictions)

    # A function that traverses the tree based on the values in the example
    def _traverse_tree(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        if row[tree['feature']] <= tree['threshold']:
            return self._traverse_tree(row, tree['left'])
        else:
            return self._traverse_tree(row, tree['right'])

