import pandas as pd
from pandas_profiling import ProfileReport

report_filename = "data_report.html"

df = pd.read_csv("framingham.csv", index_col=None)
print(df.head())

report = ProfileReport(df, interactions=None)
report.to_file(report_filename)

print(f"Wrote report to {report_filename}")
