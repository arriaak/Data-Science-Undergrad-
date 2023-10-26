import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
c_formatter = logging.Formatter('%(levelname)s@%(asctime)s: %(message)s', datefmt="%H:%M:%S")
c_handler.setFormatter(c_formatter)
logger.addHandler(c_handler)

# Read in test data
df = pd.read_csv("test.csv", index_col=False)
X = df.values[:, :-1]
Y = df.values[:, -1]
n, d = X.shape

# Read in model
with open("classifier.pkl", "rb") as pfile:
    scaler = pickle.load(pfile)
    logreg = pickle.load(pfile)

# Scale X
X_scaled = scaler.transform(X)

# Check accuracy
test_accuracy = logreg.score(X_scaled, Y)
print(f"Test Accuracy = {test_accuracy * 100.0:.1f}%")

# Show confusion matrix
Y_pred = logreg.predict(X_scaled)
cm = confusion_matrix(Y, Y_pred)
print(f"Confusion matrix with 0.5 threshold: \n{cm}")

# Get un-thresholded data
Y_fuzzy = logreg.predict_proba(X_scaled)[:, 1]

# Try a bunch of thresholds
threshold = 0.0
best_f1 = -1.0
thresholds = []
recall_scores = []
precision_scores = []
f1_scores = []

while threshold <= 1.0:
    thresholds.append(threshold)
    Y_pred = np.zeros(n, dtype=int)
    Y_pred[Y_fuzzy > threshold] = 1
    accuracy = (Y == Y_pred).sum() / n
    positives = Y_pred.sum()
    if positives == 0:
        recall = 0.0
        precision = 1.0
    elif positives == n:
        recall = 1.0
        precision = 0.0
    else:
        recall = recall_score(Y, Y_pred)
        precision = precision_score(Y, Y_pred)
    recall_scores.append(recall)
    precision_scores.append(precision)
    if recall == 0.0 or precision == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
    logger.info(
        f"Threshold={threshold:.3f} Accuracy={accuracy:.3f} Recall={recall:.2f} Precision={precision:.2f} F1 = {f1:.3f}"
    )
    threshold += 0.02

Y_pred = np.zeros(n, dtype=int)
Y_pred[Y_fuzzy > best_threshold] = 1

cm = confusion_matrix(Y, Y_pred)
print(f"Confusion matrix with {best_threshold:.3f} threshold: \n{cm}")

fig, ax = plt.subplots()
ax.plot(thresholds, recall_scores, "b", label="Recall")
ax.plot(thresholds, precision_scores, "g", label="Precision", color="g")
ax.plot(thresholds, f1_scores, "r", label="F1", color="r")
ax.vlines(best_threshold, 0, 1, "r", linewidth=0.5, linestyle="dashed")
ax.set_xlabel("Threshold")
ax.legend()
fig.savefig("threshold.png")
