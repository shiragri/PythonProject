
import pandas as pd
import numpy as np
import math

class KNNClassifier:
    def __init__(self, k=3): # k=3 is the default.
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y



    def predict(self, x, k=3):
        predictions = [] #A list that will hold the results for each row.

        for test_point in x.values:
            distances = [] #A list that will hold the results for the distances.

            for i, train_point in enumerate(self.x_train.values):  # Calculating the distance for each data point.
                distance = math.sqrt(np.sum((train_point - test_point) ** 2)) #The Euclidean distance.
                distances.append((distance,self.y_train.iloc[i]))

            # Sorting the distances and selecting the k nearest neighbors.
            distances = sorted(distances, key=lambda x: x[0]) # Sorting the list of distances in ascending order.
            k_nearest_labels = [label for _, label in distances[:k]] # Selecting the k nearest neighbors from the sorted list and taking their prediction.

           # Determining the category by majority vote.
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)

        return np.array(predictions)
