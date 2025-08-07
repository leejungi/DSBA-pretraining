"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: Main file 
	@author: JungiLee
"""
import json
import torch
import random
import numpy as np
import torchvision.transforms as transforms
#import wandb

from module.CLSModule import CLSModule
from utils.log import *
from utils.parse import parse

def augmentation(crop_size=32, padding=4, resize=32, train=True):
	norm = [
				transforms.Resize(resize),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
				]

	if train==True:
		return transforms.Compose([
#				transforms.RandomCrop(crop_size, padding=padding),
				transforms.RandomResizedCrop(crop_size, scale=(0.8,1.0)),
				transforms.RandomHorizontalFlip()]
				+norm
				)
	else:
		return transforms.Compose(norm)

def main(cfg):
#	wandb.init(project="DSBA pretraining", config=cfg)
	#fix seed for reproducibility
	if cfg['seed'] != -1:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(cfg['seed'])
		random.seed(cfg['seed'])
		torch.manual_seed(cfg['seed'])
		torch.cuda.manual_seed_all(cfg['seed'])


	#log configuration 
	make_logger(cfg['model'], cfg['device'], cfg['ckpt_dir'], cfg['log_dir'])
	log_str=f"parsing argument\n\t{'key':30} {'value':30}"
	log_str+=f"\n"+"-"*61
	log_str+=' '.join(f'\n\t{k:<30} {v}' for k, v in cfg.items())
	log(log_str)


	#set augmentation 
	if cfg['pretrained']==0 and cfg['model'].lower() != "vit":
		if cfg['dataset'].lower() in ['cifar10', 'cifar100']:
			resize=32
		else:
			resize=64
	else:
		resize=224

	if cfg['dataset'].lower() in ['cifar10', 'cifar100']:
		crop_size=32
	else:
		crop_size=64
	train_transform=augmentation(crop_size=crop_size, resize=resize)
	test_transform=augmentation(crop_size=crop_size, resize=resize, train=False)

	#Load dataset & set loader
	dataset = __import__('dataloader').__dict__[cfg['dataset']](
													root=cfg['data_path'],
													batch_size=cfg['batch'],
													num_workers=cfg['num_workers'],
													seed=cfg['seed'],
													train_transform=train_transform,
													test_transform=test_transform
													)
	dataset.setup()

	#Train model
	module = CLSModule(cfg)
	module.fit(dataset.train_dataloader())

	#Test model
	result = module.predict(dataset.test_dataloader())

	for k,v in result.items():
		log(f"{k}: {v:.3f}")
	return result


if __name__=="__main__":
	parser = parse()
	cfg = parser.parse_args()
	cfg = vars(cfg)
	main(cfg)
