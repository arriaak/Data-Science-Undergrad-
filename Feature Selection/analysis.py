import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

# Deal with command-line
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv>")
    sys.exit(1)
infilename = sys.argv[1]

# Read in the basic data frame
df = pd.read_csv(infilename, index_col="property_id")
X_basic = df.values[:, :-1]
labels_basic = df.columns[:-1]
Y = df.values[:, -1]

# Expand to a 2-degree polynomials
polytransformer = PolynomialFeatures(2)
X = polytransformer.fit_transform(X_basic)
labels = polytransformer.get_feature_names_out(labels_basic)
n, d = X.shape

# Prepare for loop
residual = Y

# We always need the column of zeros to
# include the intercept
feature_indices = [0]

regressor = LinearRegression(fit_intercept=False)
print("First time through: using original price data as the residual")

while len(feature_indices) < 3:

    # Compute the p-value for the pearson correlation for each input
    # compared to the residual (which the first time will the actual prices)
    results = [(pearsonr(X[:, i], residual)[1], i) for i in range(1, d)]

    # Sort smalled p-value first
    sorted_results = sorted(results)

    # Print the results
    for i in range(d - 1):
        pvalue, idx = sorted_results[i]
        print(f'\t"{labels[idx]}" vs residual: p-value={pvalue}')

    # Which is the best?
    best_idx = sorted_results[0][1]
    feature_indices.append(best_idx)

    # List off the attributes we will use
    # to create the next residual
    print("**** Fitting with [", end="")
    for idx in feature_indices:
        print(f'"{labels[idx]}" ', end="")
    print("] ****")

    # Fit to the selected features
    sub_X = X[:, feature_indices]
    regressor.fit(sub_X, Y)
    # Get an R2 score
    print(f"R2 = {regressor.score(sub_X, Y)}")
    prediction = regressor.predict(sub_X)

    # Update the residual for the next time
    print("Residual is updated")
    residual = Y - prediction

# Any relationship between the final residual and the unused variables?
print("Making scatter plot: age_of_roof vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:, 3], residual, marker="+")
fig.savefig("ResidualRoof.png")

print("Making a scatter plot: miles_from_school vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:, 4], residual, marker="+")
fig.savefig("ResidualMiles.png")
