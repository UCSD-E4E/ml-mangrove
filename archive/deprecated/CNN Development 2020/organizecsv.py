import pandas as pd 
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", help="results csv to be processed")
args = parser.parse_args()

if args.csv:
	input_file = args.csv

df = pd.read_csv(input_file)
df['max'] = df[['dirt','mangrove','mud','succulent','water']].idxmax(axis=1)
cwd = os.getcwd()
input_file = "max" + input_file
df.to_csv("site15resultsmax.csv")

for index, row in df.iterrows():
    cur_file = os.path.basename(row['file'])
    classification = row['max'] 
    dest = os.path.join(cwd,classification,cur_file)
    src = row['file']
    print(src)
    print(dest)
    os.rename(src, dest)

