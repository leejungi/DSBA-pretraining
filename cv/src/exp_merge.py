"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: experiment merge file
	@author: JungiLee
"""

import numpy as np
import os
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter

if __name__ =="__main__":
	parser = argparse.ArgumentParser(description="Experiments merging", formatter_class=RawTextHelpFormatter) 
	parser.add_argument("--exp_num", type=str, default="1", help="Experiment information")
	args = parser.parse_args()

	#Experiment integrate
	float_format="%3f"

	exp_dir = f"./EXP/EXP_{args.exp_num}"
	type_dirs = os.listdir(exp_dir)

	df_dict = {}
	for t in type_dirs:
		if os.path.isdir(f"{exp_dir}/{t}"):
			excel_path = f"{exp_dir}/{t}/{t}.xlsx"
			if os.path.isfile(excel_path):
				df= pd.read_excel(excel_path, sheet_name ='exp', index_col=[0], engine="openpyxl")

				for c in df.columns:
					if c not in df_dict:
						main_df = pd.DataFrame()
					else:
						main_df = df_dict[c]
					main_df[t] = df[c]
					df_dict[c] = main_df

	writer = pd.ExcelWriter(f"{exp_dir}/EXP_{args.exp_num}_result.xlsx", engine="xlsxwriter")

	sheet = list(df_dict.keys())
	del sheet[sheet.index('mean')]
	del sheet[sheet.index('var')]
	for k in ['mean', 'var'] + sheet:
		df_dict[k].to_excel(writer, sheet_name=f'{k}', float_format=float_format)
	writer.close()
