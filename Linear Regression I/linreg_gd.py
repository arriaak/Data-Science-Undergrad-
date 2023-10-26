import numpy as np
import pandas as pd
import sys
import util
import matplotlib.pyplot as plt

# Check the command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# Learning rate
t = 0.001

# Limit interations
max_steps = 1000

# Get the arg and read in the spreadsheet
infilename = sys.argv[1]
X, Y, labels = util.read_excel_data(infilename)
n, d = X.shape
print(f"Read {n} rows, {d - 1} features from '{infilename}'.")

# Get the mean and standard deviation for each column
means = X.mean(axis=0)
std = X.std(axis=0)

# Don't mess with the first column (the 1s)
means[0] = 0.0
std[0] = 1.0

# Standardize X
Xp = (X - means) / std

# First guess is "all coefficents are zero"
new_B = np.zeros(d)

# Initializing stuff for the loop
errors = np.zeros(max_steps)

for i in range(max_steps):
    # Note the current B
    current_B = new_B

    # Compute a new B
    gradient = Xp.T @ (Xp @ current_B - Y)
    new_B = current_B - t * gradient

    # Figure out the error with the new B
    prediction = Xp @ new_B
    residual = Y - prediction
    avg_error = np.dot(residual, residual) / n
    errors[i] = avg_error

    # Check to see if we have converged
    if np.sum((new_B - current_B) ** 2) < 10**-2:
        break

print(f"Took {i} iterations to converge")

# Adjust the y-intercept
means[0] = -1.0
new_B[0] = new_B @ (-1 * means / std)

# Adjust the coefficients
B = new_B / std

# Show the result
print(util.format_prediction(B, labels))

# Get the R2 score
R2 = util.score(B, X, Y)
print(f"R2 = {R2:f}")

# Draw a graph
fig1 = plt.figure(1, (4.5, 4.5))
ax1 = fig1.add_axes([0.15, 0.15, 0.7, 0.7])
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.plot(np.arange(i), errors[:i])
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Mean squared error")
ax1.set_title("Convergence")
fig1.savefig("err.png")
