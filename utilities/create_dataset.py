import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

#tabella: case_id   slide_id	 survival_months	 vital_status


PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
"""
function to create ovary cancer dataset
"""
def create_db(args):
      path_mRNA_coding: Path = Path(
            PROJECT_ROOT / 'Data' / 'mRNA_Data' / 'mRNA_table_clean.tsv')
      path_met_coding: Path = Path(
            PROJECT_ROOT / 'Data' / 'met_Data' / 'met_table_clean.tsv')
      path_OS: Path = Path(
            PROJECT_ROOT / 'Data' / 'OS_Data' / 'OS_table.tsv')
      path_WSI: Path = Path(
            PROJECT_ROOT / 'Data' / 'WSI_Data' / 'WSI_table.tsv')
      path_output: Path = Path(
            PROJECT_ROOT / 'Data' / f"tcga_ovary_all_clean_{args.met}.csv")   

      df_mRNA = pd.read_csv(path_mRNA_coding, sep='\t')
      df_OS = pd.read_csv(path_OS, sep='\t')
      df_WSI = pd.read_csv(path_WSI, sep='\t')

      gene_names = df_mRNA['gene_name'] + '_rnaseq'
      columns = ['case_id', 'slide_id', 'survival_months', 'vital_status'] 
      if args.met:
           df_met = pd.read_csv(path_met_coding, sep='\t')
           met_names = df_met['gene_name'] + '_met'
           gene_names = pd.concat([gene_names,met_names])
           case_id_met = df_met.columns[2:]#592
      columns.extend(gene_names)

      table = pd.DataFrame(columns=columns) #387 x 18945 senza met / 378 x 33000 con met  
      
      case_id_OS = df_OS.columns[1:] #587
      case_id_WSI = df_WSI['case_id'] #387
      case_id_mRNA = df_mRNA.columns[2:]#497
      common_case_ids = set(case_id_OS) & set(case_id_mRNA) & set(case_id_WSI) #387
      if args.met:
            common_case_ids = set(common_case_ids) & set(case_id_met) #378
      common_case_ids = list(common_case_ids) #387 senza met/ 378 con met

      table['case_id'] = common_case_ids 

      for i in range (len(common_case_ids)):
            case_id = common_case_ids[i]
            table.iloc[i, 4:18945] = df_mRNA[case_id]
            if args.met:
                  table.iloc[i, 18945:] = df_met[case_id]

            table.iloc[i, 0] = df_WSI[df_WSI['case_id'] == case_id]['slide_id'].iloc[0] #case_id = slide_id
            table.iloc[i, 1] = df_WSI[df_WSI['case_id'] == case_id]['file'].iloc[0]
            vital_status=df_OS[case_id][0]
            if vital_status=='Dead': #215
                  table.iloc[i, 2] = df_OS[case_id][1] #'survival_months'
                  table.iloc[i, 3] = 0 
            else: #Alive = 170
                  table.iloc[i, 2] = df_OS[case_id][2] #'survival_months'
                  table.iloc[i, 3] = 1 

      table_unique_columns = table.loc[:, ~table.columns.duplicated()]
      table_unique_columns = table_unique_columns.dropna()
      table_unique_columns.to_csv(path_output, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--met", default=True, type=bool) #whether or not to construct the dataset with also methylation values
    args = parser.parse_args()
    create_db(args)



