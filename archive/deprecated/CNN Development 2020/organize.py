import pandas as pd 
import os 
df = pd.read_csv("site1resultsmax.csv")
df['max'] = df[['dirt','mangrove','mud','succulent','water']].idxmax(axis=1)
cwd = os.getcwd()

for index, row in df.iterrows():
    cur_file = os.path.basename(row['file'])
    classification = row['max'] 
    dest = os.path.join(cwd,classification,cur_file)
    src = row['file']
    print(src)
    print(dest)
    os.rename(src, dest)

