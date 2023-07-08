from tkinter.font import Font

import numpy as np
import tkinter as tk
from tkinter import ttk

from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from collections import Counter
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.3
K = 3


class NN:

    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        # FEATURE= X_TEST
        # K= number of neighbours
        predicted_labels = []
        for feature in features:
            distances = [euclidean(feature, train_feature) for train_feature in self.trainingFeatures]
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = np.array(self.trainingLabels)[nearest_indices]
            majority_vote = Counter(nearest_labels).most_common(1)
            predicted_labels.append(majority_vote[0][0])
        return predicted_labels


def load_data(filename):
    features = []
    labels = []

    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            feature_vector = list(map(float, values[:57]))
            target_label = int(values[57])
            features.append(feature_vector)
            labels.append(target_label)

    return features, labels


def preprocess(features):
    """
    normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation
    """
    normalized_features = np.array(features)

    for i in range(normalized_features.shape[1]):  # Iterate over each feature (column)
        column_values = normalized_features[:, i]
        mean = np.mean(column_values)
        std = np.std(column_values)

        if std == 0:
            # Handle columns with zero standard deviation
            normalized_features[:, i] = column_values
            print("Standard deviation is zero, cannot perform normalization.")
            print("-------------------------------------------\n")

        else:
            # Perform normalization for non-zero standard deviation
            for index in range(len(column_values)):  # update the values for each column
                column_values[index] = (column_values[index] - mean) / std
            normalized_features[:, i] = column_values  # set the updated values in the normalized features array
            # print("AFTER NORMALIZATION:", normalized_features[:, i])
            # print("-------------------------------------------\n")
    return normalized_features


def train_mlp_model(features, labels):
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500)  # Set max_iter to a higher value, e.g., 500
    mlp.fit(features, labels)
    return mlp


def evaluate(labels, predictions):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index in range(len(labels)):
        if labels[index] == predictions[index] and labels[index] == 1:
            TP = TP + 1
        elif labels[index] == predictions[index] and labels[index] == 0:
            TN = TN + 1
        elif labels[index] != predictions[index]:
            if labels[index] == 1:
                FN = FN + 1
            else:
                FP = FP + 1

    print("TP= ", TP)
    print()
    print("FP= ", FP)
    print()
    print("TN= ", TN)
    print()
    print("FN= ", FN)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1


def create_table(window, data, y_test, predictions, type):
    style = ttk.Style()
    style.configure("Custom.Treeview.Heading", background="red")

    table = ttk.Treeview(window)
    table["columns"] = ("Actual Label", "Predicted Label")
    table.column("#0", width=0, stretch=tk.NO)
    table.column("Actual Label", anchor=tk.CENTER, width=100)
    table.column("Predicted Label", anchor=tk.CENTER, width=100)

    bold_font = Font(family="Arial", size=12, weight="bold")

    table.heading("#0", text="")
    table.tag_configure("Custom.Treeview.Heading", background="red")
    table.heading("Actual Label", text="Actual Label", anchor=tk.CENTER)
    table.heading("Predicted Label", text="Predicted Label", anchor=tk.CENTER)

    table.tag_configure("spam", foreground="red")
    table.tag_configure("spam2", foreground="green")

    for index, (actual, predicted) in enumerate(data):
        if actual == 1 and predicted == 1:
            table.insert("", index, text="", values=("Spam", "Spam"), tags="spam2")
        elif actual == 1 and predicted == 0:
            table.insert("", index, text="", values=("Spam", "Not Spam"), tags="spam")
        elif actual == 0 and predicted == 0:
            table.insert("", index, text="", values=("Not Spam", "Not Spam"), tags="spam2")
        elif actual == 0 and predicted == 1:
            table.insert("", index, text="", values=("Not Spam", " Spam"), tags="spam")

    table.pack(fill=tk.BOTH, expand=True)  # Make the table fill the window


def show_table(y_test, predictions, type):  # TO CREATE A TABLE INCLUDING THE LABELS AND THE PREDICTED LABELS
    # Create the GUI window
    window = tk.Tk()
    window.title("K_NN Comparison Table")
    window.state('zoomed')  # Maximize the window
    window.geometry("300x200")

    # Create the title label
    title_font = Font(family="Arial", size=16, weight="bold")
    title_label = tk.Label(window, text=type, font=title_font, fg="black")
    title_label.pack(pady=10)

    data = list(zip(y_test, predictions))

    # Create the table
    create_table(window, data, y_test, predictions, type)

    # Start the GUI event loop
    window.mainloop()


def main():
    index = 1
    # call method to read the samples file
    features, labels = load_data("samples.txt")
    combined_data = list(zip(features, labels))
    # TO PRINT FILE DATA
    """ for feature, label in combined_data:
        print("Features[", str(index), "]", feature)
        print("Label:", label)
        index = index+1
        print()"""
    # Check command-line arguments
    """if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")"""

    features = preprocess(features)  # NORMALIZING THE FEATURES

    # split DATA into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    print()  # Print a new line after all predictions are printed
    print("THE ACTUAL LABEL OF THE TESTING SET: ", y_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** 1-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("------------------------------------------------")
    show_table(y_test, predictions, "K-NN")

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    print()  # Print a new line after all predictions are printed
    print("THE ACTUAL LABEL OF THE TESTING SET: ", y_test)
    print("----------------------------------------------------")
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    show_table(y_test, predictions, "MLP")


if __name__ == "__main__":
    main()
