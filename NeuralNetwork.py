import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Creating a Neural Network from Scratch
class NeuralNetwork:
    def __init__(self):
        self.weights = np.ones(1)
        self.bias = 0
        self.cost = 0
        self.epoch_list = None
        self.cost_list = None

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def logloss(self, y_truth, y_pred):
        epsilon = 1e-15
        y_pred_new = [max(i, epsilon) for i in y_pred]
        y_pred_new = [min(i, 1 - epsilon) for i in y_pred_new]
        y_pred_new = np.clip(y_pred_new, epsilon, 1 - epsilon)

        return -np.mean(
            y_truth * np.log(y_pred_new) + (1 - y_truth) * np.log(1 - y_pred_new)
        )

    def mse(self, y_truth, y_predict):
        return np.mean(np.square(y_truth - y_predict))

    def plot(self):
        epoch_list = self.epoch_list
        cost_list = self.cost_list
        plt.title("Epochs vs Cost chart")
        plt.plot(epoch_list, cost_list)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.show()

    def fit(
        self,
        X,
        y,
        optimizer=None,
        size=None,
        epoch=500,
        learning_rate=0.1,
        verbose=True,
    ):
        # try:
        if optimizer == "batch":
            self.weights, self.bias, self.cost, self.epoch_list, self.cost_list = (
                self.batch_gradient_descent(X, y, epoch, learning_rate, verbose)
            )
        elif optimizer == "mini":
            if size is None:
                raise ValueError(
                    "For mini-batch gradient descent, 'size' must be specified."
                )
            self.weights, self.bias, self.cost, self.epoch_list, self.cost_list = (
                self.mini_batch_gd(X, y, size, epoch, learning_rate, verbose)
            )
        elif optimizer == "sgd":
            self.weights, self.bias, self.cost, self.epoch_list, self.cost_list = (
                self.sgd(X, y, epoch, learning_rate, verbose)
            )
        else:
            # Default to batch gradient descent if optimizer is not specified
            self.weights, self.bias, self.cost, self.epoch_list, self.cost_list = (
                self.batch_gradient_descent(X, y, epoch, learning_rate)
            )
        print(
            f"Final Weights: {self.weights}, Final Bias: {self.bias}, Final Cost: {self.cost}"
        )

    def predict(self, X):
        X_array = np.array(X)

        # Weighted sum
        weighted_sum = np.dot(self.weights, X_array.T) + self.bias

        # Apply Activation function
        y_pred = self.sigmoid(weighted_sum)

        return y_pred

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        convert = [1 if i > 0.5 else 0 for i in y_pred]
        correct_pred = np.sum(
            [1 if pred == truth else 0 for pred, truth in zip(convert, y_test)]
        )

        return correct_pred / len(y_test)

    def batch_gradient_descent(self, X, y, epoch, learning_rate, verbose):
        num_of_features = X.shape[1]
        weights = np.ones(num_of_features)
        bias = 0
        num_of_samples = X.shape[0]

        epoch_list = []
        cost_list = []

        for i in range(epoch):
            # Weighted sum
            weighted_sum = np.dot(X, weights) + bias

            # Apply Activation function
            y_pred = self.sigmoid(weighted_sum)

            # Calculate Gradient of Weights and Bias
            weights_gradient = -(2 / num_of_samples) * np.dot(X.T, (y - y_pred))
            bias_gradient = -(2 / num_of_samples) * np.sum(y - y_pred)

            # Update Weights and Bias
            weights -= learning_rate * weights_gradient
            bias -= learning_rate * bias_gradient

            # Calculate Cost

            cost = self.logloss(y, y_pred)

            if i % 10 == 0:
                if verbose:
                    print(f"Epoch: {i}, Weights: {weights}, Bias: {bias}, Cost: {cost}")
                epoch_list.append(i)
                cost_list.append(cost)

        return weights, bias, cost, epoch_list, cost_list

    def mini_batch_gd(self, X, y, size, epoch, learning_rate, verbose):
        num_of_features = X.shape[1]
        weights = np.ones(num_of_features)
        bias = 0
        num_of_samples = len(X)
        num_of_batches = num_of_samples // size

        epoch_list = []
        cost_list = []

        for i in range(epoch):
            indices = np.arange(num_of_samples)
            np.random.shuffle(indices)

            for batch in range(num_of_batches):
                batch_indices = indices[batch * size : (batch + 1) * size]
                if isinstance(X, pd.DataFrame):
                    X_batch = X.iloc[batch_indices].values
                    y_batch = y.iloc[batch_indices].values
                else:
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]

                y_pred = np.dot(X_batch, weights) + bias
                y_pred = self.sigmoid(y_pred)

                weights_gradient = -(2 / size) * np.dot(X_batch.T, (y_batch - y_pred))
                bias_gradient = -(2 / size) * np.sum(y_batch - y_pred)

                weights -= learning_rate * weights_gradient
                bias -= learning_rate * bias_gradient

            y_pred_all = self.sigmoid(np.dot(X, weights) + bias)
            cost = self.logloss(y, y_pred_all)

            if i % 10 == 0:
                if verbose:
                    print(f"Epoch: {i}, Weights: {weights}, Bias: {bias}, Cost: {cost}")
                epoch_list.append(i)
                cost_list.append(cost)

        return weights, bias, cost, epoch_list, cost_list

    def sgd(self, X, y, epoch, learning_rate, verbose):
        num_of_features = X.shape[1]
        weights = np.ones(num_of_features)
        bias = 0
        num_of_samples = len(X)

        epoch_list = []
        cost_list = []

        for i in range(epoch):
            index = random.randint(0, num_of_samples - 1)
            if isinstance(X, pd.DataFrame):
                random_X = X.iloc[index].values
                random_y = y.iloc[index]
            else:
                random_X = X[index]
                random_y = y[index]

            y_pred = np.dot(random_X, weights) + bias

            weights_gradient = -(2 / num_of_samples) * np.dot(
                random_X.T, (random_y - y_pred)
            )
            bias_gradient = -(2 / num_of_samples) * np.sum(random_y - y_pred)

            weights -= learning_rate * weights_gradient
            bias -= learning_rate * bias_gradient

            cost = np.square(random_y - y_pred)

            if i % 10 == 0:
                if verbose:
                    print(f"Epoch: {i}, Weights: {weights}, Bias: {bias}, Cost: {cost}")
                epoch_list.append(i)
                cost_list.append(cost)

        return weights, bias, cost, epoch_list, cost_list


if __name__ == "__main__":

    # # Loading the data
    df = pd.read_csv("datas//processed_titanic.csv")

    # X and y split
    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]

    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    myNN = NeuralNetwork()

    # Training Neural Network
    myNN.fit(
        X_train,
        y_train,
        optimizer="mini",
        size=20,
        epoch=1000,
        learning_rate=0.1,
        verbose=False,
    )

    # Prediction
    predict = myNN.predict(X_test)
    predict = [1 if i > 0.5 else 0 for i in predict]

    # Score
    print(myNN.score(X_test, y_test))

    print(f"Total num of Predictions: {len(predict)}")
    print(f"Num of wrong Predictions: {np.sum(y_test != predict)}")

    myNN.plot()
