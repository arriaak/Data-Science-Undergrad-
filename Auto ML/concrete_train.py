import pycaret.regression as carreg
from time import perf_counter
import pandas as pd

# How many models we will tune
n_select = 6

# Read in the training data
train_df = pd.read_csv("train_concrete.csv", index_col=False)

# Set up the regression experiment session
print("*** Setting up session***")
t1 = perf_counter()
exp1 = carreg.setup(data=train_df, target="csMPa")
t2 = perf_counter()
print(f"*** Set up: {t2 - t1:.2f} seconds")

# Do a basic comparison of models
best = carreg.compare_models(turbo=False, n_select=6)
t3 = perf_counter()
print(f"*** compare_models: {t3 - t2:.2f} seconds")

# List the best models
print(f"*** Best:")
for b in best:
    print(f"\t{b.__class__.__name__}")

# Go through the list of models
for i, model in enumerate(best):
    label = f"{model.__class__.__name__}"
    print(f"\n\n*** {i} - {label} ***")

    # Tune the model (try 24 parameter combinations)
    tuned_model = carreg.tune_model(model, n_iter=24)

    # Finalize the model
    finalized_model = carreg.finalize_model(tuned_model)

    # Save the model
    carreg.save_model(finalized_model, label, verbose=False)

t4 = perf_counter()
print(f"*** Tuning and finalizing: {t4 - t3:.2f} seconds")
print(f"*** Total time: {t4 - t1:.2f} seconds")
