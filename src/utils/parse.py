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
		parser = argparse.ArgumentParser(description="DSBA Pretraining template code", formatter_class=RawTextHelpFormatter)

	#Parsing Argument 
	#Environment
	parser.add_argument("--dataset", type=str, default="CIFAR10", help="Dataset")
	parser.add_argument("--data_path", type=str, default="/workspace/code/data", help="Dataset")
	parser.add_argument("--device", type=str, default="cuda:0", help="Device for torch")
	parser.add_argument("--ckpt_dir", type=str, default="./ckpt", help="Directory path for ckeckpoint")
	parser.add_argument("--log_dir", type=str, default="./log", help="Directory path for log")
	parser.add_argument("--seed", type=int, default=-1, help="Seed for reproducing the result")
	parser.add_argument("--num_workers", type=int, default=4, help="The number of workers")

	#Hyperparameter
	parser.add_argument("--epoch", type=int, default=100, help="The number of epochs")
	parser.add_argument("--batch", type=int, default=256, help="The number of batch size")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
	parser.add_argument("--wd", type=float, default=0.00000, help="Weight decay")

	#Model
	parser.add_argument("--model", type=str, choices=['','R18', 'R34', 'R50', 'vit'], default="R18", help="Model type")
	parser.add_argument("--optimizer", type=str, default="Adam", help="Set Optimizer")
	parser.add_argument("--scheduler", type=str, default='cosine', help="Set scheduler")
	parser.add_argument("--pretrained", type=int, default=0, help="Flag for pretrained model\n"
																	"0: training with initialization\n"
																	"1: Finetuning\n"
																	"2: Finetuning only classifier\n"
																)

	return parser
