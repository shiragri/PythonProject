import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.prior_probs = {}
        self.conditional_probs = {}

    def fit(self, x, y):

        self.classes = np.unique(y) # Identifies the possible outcomes in prediction.


    #Calculating the probability that the loan will be approved and the probability that it will not be approved.
        self.prior_probs = {}
        for cls in self.classes:
            class_probability = np.mean(y == cls)
            log_probability = np.log(class_probability)
            self.prior_probs[cls] = log_probability

        self.conditional_probs = {}
        for cls in self.classes:
            class_data = x[y == cls] # Divides the data into rows that were approved or not approved.
            self.conditional_probs[cls] = {}
            for col in x.columns: # For each column in the data, we calculate the conditional probabilities for the different values in that column.
                self.conditional_probs[cls][col] = self._calculate_conditional_probs(class_data[col])


    # A function that calculates the conditional probabilities for a specific column with Laplace Smoothing and log.
    def _calculate_conditional_probs(self, col):

        unique_values = np.unique(col)
        total_count = len(col)
        num_unique_values = len(unique_values)
        probs = {}
        for val in unique_values:
            probs[val] = np.log((np.sum(col == val) + 1) / (total_count + num_unique_values))
        return probs

    def predict(self, x):

        predictions = []
        for _, row in x.iterrows():
            class_probs = {} # For a loan that was approved or not approved, the model calculates the overall probability.
            for cls in self.classes:
                prob = self.prior_probs[cls]
                for col in x.columns:
                    value = row[col]

                    prob += self.conditional_probs[cls][col].get( # We use addition because adding logarithms is like multiplying.
                        value,
                        np.log(1 / (len(x) + len(self.conditional_probs[cls][col])))  # Laplace Smoothing
                    )
                class_probs[cls] = prob
            predictions.append(max(class_probs, key=class_probs.get)) # Selects the category with the highest probability.
        return np.array(predictions)

