import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import kstest, norm
import matplotlib.pyplot as plt
import sys
import util


# Check command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# Read in the argument
infilename = sys.argv[1]

# Read the spreadsheet
X, Y, labels = util.read_excel_data(infilename)

n, d = X.shape
print(f"Read {n} rows, {d-1} features from '{infilename}'.")

# Don't need the intercept added -- X has column of 1s
lin_reg = LinearRegression(fit_intercept=False)

# Fit the model
lin_reg.fit(X, Y)

# Pretty print coefficients
print(util.format_prediction(lin_reg.coef_, labels))

# Make predictions
predictions = lin_reg.predict(X)

# Get residual
residual = Y - predictions

# Make a histogram of the residuals
fig1 = plt.figure(1, (4.5, 4.5))
ax1 = fig1.add_axes([0.2, 0.12, 0.7, 0.8])
ax1.hist(residual, bins=18)
ax1.set_xlabel("Residual")
ax1.xaxis.set_major_formatter(lambda x, pos: f"${x/1000:.0f}K")
ax1.set_ylabel("Density")
ax1.set_title("Residual Histogram")
fig1.savefig("res_hist.png")

# Check to see if it is normal
result = kstest(residual, norm.cdf)
print(f"Kolmogorov-Smirnov: P-value = {result.pvalue}")
if (result.pvalue < 0.05):
    print(f"\tThe residual follows a normal distribution.")
else:
    print(f"\The residual does not follow a normal distribution.")
    sys.exit(0)

# Compute the variance (remembering to 
# compensate for the degrees of freedom)
variance = residual @ residual / (n - d - 1)
standard_deviation = np.sqrt(variance)
print(f"68% of predictions with this formula will be within ${standard_deviation:,.02f} of the actual price.")
print(f"95% of predictions with this formula will be within ${2.0 * standard_deviation:,.02f} of the actual price.")

