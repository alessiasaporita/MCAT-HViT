import json
import pandas as pd
from pathlib import Path
import numpy as np
import json
import pandas as pd
from pathlib import Path
import numpy as np
import argparse

"""
function to clean mRNA and methylation table by filtering out for protein coding genes and applying log2 transformation to mRNA values
"""
#tabella: CASE_ID
#1
...
#27.000/60600

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()

def filter_table(args):
    path: Path = Path(
        PROJECT_ROOT / 'gene_with_protein_product.json')
    path_table: Path = Path(
          PROJECT_ROOT / 'Data' / f"{args.data_type}_Data" / f"{args.data_type}_table.tsv")
    path_table_coding: Path = Path(
          PROJECT_ROOT / 'Data' / f"{args.data_type}_Data" / f"{args.data_type}_table_clean.tsv")

    #Filtering for protein-coding gene
    with open(path, 'r', encoding='utf-8') as f:
        file_json = json.load(f)

    protein_id = [] 

    for doc in file_json["response"]["docs"]:
        protein_id.append(doc["symbol"])
    
    df = pd.read_csv(path_table, delimiter='\t')
    filtered_df = df[df['gene_name'].isin(protein_id)]
    filtered_df = filtered_df.reset_index().drop(columns='index')

    #Normalization 
    if args.norm and args.data_type == 'mRNA': #mRNA, met
        table_norm = filtered_df.iloc[:,2:]
        epsilon = 0.01
        table_norm = np.log2(table_norm+epsilon)
        #scaler = StandardScaler()
        #table_norm = pd.DataFrame(scaler.fit_transform(table_norm), columns = table_norm.columns)
        filtered_df.iloc[:,2:] = table_norm
    
    filtered_df = filtered_df.dropna()
    filtered_df.to_csv(path_table_coding, sep='\t', index=False) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default='met', choises = ['met', 'mRNA'], type=str) 
    parser.add_argument("--norm", default=True, type=bool)
    args = parser.parse_args()
    filter_table(args)


