"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: experiment file
	@author: JungiLee
"""

import os
import numpy as np
import sys
import argparse
from argparse import RawTextHelpFormatter
import json
import subprocess
import pandas as pd

from main import main
from utils.parse import parse

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description="Experiment", formatter_class=RawTextHelpFormatter)
	parser.add_argument("--exp_num", type=str, default="1", help="Experiment Directory")
	parser.add_argument("--case_name", type=str, default="noname", help="Experiment Case folder name")

	parser=parse(parser)
	cfg = parser.parse_args()
	cfg = vars(cfg)

	float_format="%3f"


	def make_dir(path):
		path = path.split("/")

		Dir = path[0] 

		for p in path[1:]:
			Dir += f"/{p}"
			if not os.path.isdir(Dir):
				os.mkdir(Dir)

	f_name=cfg['case_name']
	exp_dir = f"./EXP/EXP_{cfg['exp_num']}/{f_name}"
	make_dir(exp_dir)

	seed_dir = exp_dir + f"/{cfg['seed']}"
	if os.path.isdir(seed_dir):
		cmd = f"rm -r {seed_dir}"
		subprocess.call(cmd, shell=True)
	os.mkdir(seed_dir)

	cfg_index = list(cfg.keys())
	result_dict = main(cfg)

	eval_index = ['accuracy', 'inference', 'loss_total', 'loss_mean']


	index = cfg_index+eval_index

	cmd = f"mv ./ckpt/'{cfg['model']}_{cfg['device']}'/info.log {seed_dir}"
	subprocess.call(cmd, shell=True)
	cmd = f"mv ./ckpt/'{cfg['model']}_{cfg['device']}'/model.pth {seed_dir}"
	subprocess.call(cmd, shell=True)

	try:
		cmd = f"mv ./ckpt/'{cfg['model']}_{cfg['device']}'/*.png {seed_dir}"
		subprocess.call(cmd, shell=True)
	except:
		pass


	df=pd.DataFrame(index=index)

	info_list =[]
	for i in cfg_index:
		info_list.append(cfg[i])

	for i in eval_index:
		if type(result_dict[i])==list or type(result_dict[i])==np.ndarray:
			print(i, result_dict[i], result_dict[i][-1])
			info_list.append(result_dict[i][-1])
		else:
			print(i, result_dict[i])
			info_list.append(result_dict[i])
		
	df[f"{cfg['seed']}"] = info_list

	writer = pd.ExcelWriter(seed_dir+f"/{cfg['seed']}.xlsx", engine='xlsxwriter')
	df.to_excel(writer, sheet_name = f"{cfg['seed']}", float_format=float_format)
	writer.close()

	#Seed integrate
	seed_dirs = sorted(os.listdir(exp_dir))
	df= pd.DataFrame(index=index)
	for seed in seed_dirs:
		if os.path.isdir(exp_dir+"/"+seed):
			excel_path = f"{exp_dir}/{seed}/{seed}.xlsx"
			if os.path.isfile(excel_path):
				tmp_df = pd.read_excel(excel_path, sheet_name=seed, index_col=[0], engine="openpyxl")
				df = pd.concat([df, tmp_df],axis=1)

	mean_list = [] 
	var_list = [] 


	for i in index:

		try:
			mean = df.loc[i].mean()
			var = df.loc[i].var()
		except:
			try:
				var=mean = ''.join(np.unique(df.loc[i]))
			except:
				var=mean = np.unique(df.loc[i])

		mean_list.append(mean)
		var_list.append(var)
	df['mean'] = mean_list
	df['var'] = var_list
	df['var'] = df['var'].fillna(0)
	writer = pd.ExcelWriter(f"{exp_dir}/{f_name}.xlsx", engine='xlsxwriter')
	df.to_excel(writer, sheet_name=f"exp", float_format=float_format)
	writer.close()
