import csv
import numpy as np
import pandas as pd
from scipy.stats import chi2

# Converts a numpy array into a list of strings
# formatted as percentages
def make_list_of_percent_strings(a):
    result = []
    for v in a:
        result.append(f"{v * 100.0:.1f}\\%")
    return result


# Creates a latex table
def latex_table(
    data, row_labels, col_labels, row_agg, col_agg, all_agg, is_percent=False
):
    is_int = data.dtype == int

    (rcount, ccount) = data.shape
    string = "\\begin{tabular}{ r | "
    for _ in range(ccount):
        string += " c "
    string += "| c }\n "
    for col_label in col_labels:
        string += f"{col_label} & "
    string += "\\\\\n"
    string += "\\hline\n"
    for row_index in range(rcount):
        string += f"{row_labels[row_index]} & "
        for col_index in range(ccount):
            v = data[row_index, col_index]
            if is_percent:
                string += f"{v * 100.0:.1f}\\% & "
            else:
                if is_int:
                    string += f"{v} & "
                else:
                    string += f"{v:.1f} & "
        string += f"\\textbf{{{row_agg[row_index]}}} \\\\\n"
    string += "\\hline\n & "
    for col_index in range(ccount):
        string += f"\\textbf{{{col_agg[col_index]}}} & "
    string += f"\\textbf{{{all_agg}}}\n"
    string += "\\end{tabular}"
    return string


# Read in the data
mice_df = pd.read_csv("mice.csv")

# Figure out the possible gene types
gene_types = list(mice_df.gene_type.unique())
print(f"Possible gene types:{gene_types}")

# Make an dictionary to hold counts
counts_array = np.zeros((len(gene_types), 2), dtype=int)

# Count the combinations
for _, row in mice_df.iterrows():
    gene_index = gene_types.index(row["gene_type"])
    if row["has_cancer"] == True:
        cancer_index = 1
    else:
        cancer_index = 0
    counts_array[gene_index, cancer_index] += 1

# Make sums
gene_sums = np.sum(counts_array, axis=1)
cancer_sums = np.sum(counts_array, axis=0)
all_count = np.sum(cancer_sums)

# Create a table
column_labels = ["Gene", "No Cancer", "Has Cancer"]
table_str = latex_table(
    counts_array, gene_types, column_labels, gene_sums, cancer_sums, all_count
)
print(table_str)

# Make proportions
proportions = (counts_array.T / gene_sums).T
gene_proportions = gene_sums / all_count
cancer_proportions = cancer_sums / all_count

# Create a table
cancer_prop_list = make_list_of_percent_strings(cancer_proportions)
gene_prop_list = make_list_of_percent_strings(gene_proportions)
table_str = latex_table(
    proportions,
    gene_types,
    column_labels,
    gene_prop_list,
    cancer_prop_list,
    "",
    is_percent=True,
)
print(table_str)

# Create expected counts
expected_counts = np.outer(gene_sums, cancer_proportions)

# Create a table
table_str = latex_table(
    expected_counts, gene_types, column_labels, gene_sums, cancer_prop_list, ""
)
print(table_str)

# Calculate X2
x2 = np.sum(np.square(counts_array - expected_counts) / expected_counts)
print(f"X2 = {x2:.4f}")

# Calculate the p-value
p_less = chi2.cdf(x2, 2)
p = 1.0 - p_less
print(f"p-value = {p}")
