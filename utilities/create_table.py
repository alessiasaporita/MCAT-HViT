import json
import pandas as pd
import os
from pathlib import Path
import xmltodict
from argparse import ArgumentParser

#tabella: CASE_ID
#1
...
#27.000/60600/WSI_FILENAME

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
"""
function to create raw mRNA, survival, WSIs and methylation table 
"""
def create_table(args):
  path_table: Path = Path(
          PROJECT_ROOT / 'Data' / f"{args.data_type}_Data" / f"{args.data_type}_table.tsv")
  path_json: Path = Path(
          PROJECT_ROOT / 'Data' / f"{args.data_type}_Data" / f"files_{args.data_type}.json")
  data_path: Path = Path(
          PROJECT_ROOT / 'Data' / f"{args.data_type}_Data")
  path_filenames: Path = Path(
          PROJECT_ROOT / 'Data' / f"WSI_Data" / 'file_names.txt')
    
  file_names_list = []
  with open(path_filenames, 'r') as f:
    file_names_list = f.read().splitlines() #387
  
  with open(path_json, 'r') as f:
    file_json = json.load(f)

  case_id_list = [] #lista case_id univoci

  for dict in file_json:
    case_id_list.append(dict["cases"][0]["case_id"])

  case_id_list=list(set(case_id_list))
  table = pd.DataFrame()

  if args.data_type == 'met' or args.data_type == 'mRNA': 
    df_path: Path = Path(
          PROJECT_ROOT / 'Data' / 'illumina_humanmethylation27_content.xlsx')
    df_genename = pd.read_excel(df_path, header=0)
    table["name"] = df_genename.iloc[:, 0]
    table["gene_name"] = df_genename.iloc[:, 10]
  
  if args.data_type == 'OS':
    table['info']=['vital_status', 'days_to_death', 'days_to_last_followup']

  for case_id in case_id_list:
    for i in range(len(file_json)):
      if (file_json[i]["cases"][0]["case_id"] == case_id): 
        filename = file_json[i]["file_name"]
        if args.data_type == 'WSI':
            if filename in file_names_list:
              new_row = {'case_id': case_id, 'slide_id': filename[:12], 'file': filename}
              table = table.append(new_row, ignore_index=True)
        else:
          path_datafile = os.path.join(data_path, 'Data', filename)
          if os.path.isfile(path_datafile):
            if args.data_type == 'met' or args.data_type == 'mRNA':
              df = pd.read_csv(path_datafile, header=None, delimiter='\t')
              if args.data_type == 'met':
                table[case_id] = df.iloc[:,1]
              elif args.data_type == 'mRNA':
                table[case_id] = df['fpkm_uq_unstranded']
            else: #OS
              with open(path_datafile, 'r') as f:
                data = f.read()
              json_data = json.dumps(xmltodict.parse(data), indent=2)
              path_json_data = os.path.join(data_path, 'tmp.json')
              with open(path_json_data, 'w') as f:
                f.write(json_data)
              with open(path_json_data, 'r') as f:
                json_data = json.load(f)
              field_to_check = "#text"
              if "ov:tcga_bcr" in json_data: #OV Project
                  if field_to_check in json_data["ov:tcga_bcr"]["ov:patient"]["clin_shared:vital_status"]:
                      vs = json_data["ov:tcga_bcr"]["ov:patient"]["clin_shared:vital_status"]["#text"]
                  else:
                      vs=None
                  if field_to_check in json_data["ov:tcga_bcr"]["ov:patient"]["clin_shared:days_to_death"]:                       
                      dtd=json_data["ov:tcga_bcr"]["ov:patient"]["clin_shared:days_to_death"]["#text"]
                      if json_data["ov:tcga_bcr"]["ov:patient"]["clin_shared:days_to_death"]['@precision'] == 'day':
                        dtd = round(int(dtd)/31, 2)
                  else: 
                      dtd = None
                  if field_to_check in json_data["ov:tcga_bcr"]["ov:patient"]["clin_shared:days_to_last_followup"]:
                      dtlf=json_data["ov:tcga_bcr"]["ov:patient"]["clin_shared:days_to_last_followup"]["#text"]
                      if json_data["ov:tcga_bcr"]["ov:patient"]["clin_shared:days_to_last_followup"]['@precision'] == 'day':
                        dtlf = round(int(dtlf)/31, 2)
                  else: 
                      dtlf=None
              table[case_id]= [vs, dtd, dtlf]

  table.to_csv(path_table, sep='\t', index=False)
  print("Table created")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_type", default='WSI', type=str, choices=['mRNA', 'OS', 'met', 'WSI']) 
    args = parser.parse_args()
    create_table(args)



