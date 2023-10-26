import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler


# Read in the data
print("Reading input...")
df = pd.read_csv("train.csv", index_col=False)
X = df.values[:, :-1]
Y = df.values[:, -1]

# Standardize X
print("Scaling...")
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Do logistic regresion
print("Fitting...")
logreg = LogisticRegression()
logreg.fit(X_scaled, Y)

# Check accuracy on training data
train_accuracy = logreg.score(X_scaled, Y)
print(f"Training accuracy = {train_accuracy}")

# Save scaling and logistic regression coefficients to a pickle file
pickle_path = "classifier.pkl"
print(f"Writing scaling and logistic regression model to {pickle_path}...")
with open(, "wb") as pfile:
    pickle.dump(scaler, pfile)
    pickle.dump(logreg, pfile)
