"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: Parser for hyperparamters
	@author: JungiLee
"""

import argparse
from argparse import RawTextHelpFormatter


def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def list_of_ints(arg):
	return list(map(int, arg.split(',')))
 
def parse(parser=None):
	if parser==None:
		parser = argparse.ArgumentParser(description="DSBA Pretraining NLP template code", formatter_class=RawTextHelpFormatter)

	#Parsing Argument 
	#Environment
	parser.add_argument("--device", type=str, default="cuda:0", help="Device for torch")
	parser.add_argument("--seed", type=int, default=42, help="Seed for reproducing the result")
	parser.add_argument("--max_length", type=int, default=128, help="Maximum token size")

	#Hyperparameter
	parser.add_argument("--epoch", type=int, default=5, help="The number of epochs")
	parser.add_argument("--batch", type=int, default=8, help="The number of batch size")
	parser.add_argument("--optim", type=str, default='adam', help="Optimier type")
	parser.add_argument("--lr", type=float, default=5e-5, help="Learning Rate")
	parser.add_argument("--wd", type=float, default=0, help="Weight Decay")
	parser.add_argument("--accum_step", type=int, default=1, help="Gradient Accumulation step")

	#Model
	parser.add_argument("--model", type=str, choices=['bert','modernbert'], default="bert", help="Model type")
	parser.add_argument("--ckpt_path", type=str, default="./ckpt", help="Model save path")
	return parser
