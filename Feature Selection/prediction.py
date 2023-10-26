import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv>")
    sys.exit(1)

infilename = sys.argv[1]

print("Making new features...")
df = pd.read_csv(infilename, index_col="property_id")
df["lot_size"] = df["lot_width"] * df["lot_depth"]
df["is_close_to_school"] = 0
df.loc[df["miles_to_school"] < 2, "is_close_to_school"] = 1

labels = ["sqft_hvac", "lot_size", "is_close_to_school"]
print(f"Using only the useful ones: {labels}...")

X = df[labels].to_numpy()
Y = df["price"].to_numpy()

regressor = LinearRegression()
regressor.fit(X, Y)
print(f"R2 = {regressor.score(X, Y):.5f}")
coefs = regressor.coef_.copy()
print("*** Prediction ***")
print(
    f"Price = ${regressor.intercept_:,.2f} + (sqft x ${coefs[0]:.02f}) + (lot_size x ${coefs[1]:.02f})"
)
print(
    f"\tLess than 2 miles from a school? You get ${coefs[2]:,.02f} added to the price!"
)
