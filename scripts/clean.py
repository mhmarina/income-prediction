import pandas as pd

outfile = "../data/adult.data.clean.csv"
infile = "../data/adult.data.csv"

df = pd.read_csv(infile, header=None, index_col=False)

init = df.count()
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

mask = df.eq(' ?').any(axis=1)
df = df[~mask]

final = df.count()
print(f"dropped {init - final} rows")

df.to_csv(outfile, header=None, index=False)
