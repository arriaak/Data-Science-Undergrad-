import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("framingham.csv", index_col=False)
print(f"Read {df.shape[0]} rows")

df.dropna(inplace=True)

print(f"Using {df.shape[0]} rows")

split_list = train_test_split(df, test_size=0.2)
frametypes = ["train", "test"]

for i in range(2):
    outdf = split_list[i]
    frametype = frametypes[i]
    csv_filename = f"{frametype}.csv"
    print(f"*** Writing {outdf.shape} to {frametype}")
    outdf.to_csv(csv_filename)
