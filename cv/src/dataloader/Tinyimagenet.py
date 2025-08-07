"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: Tinyimagenet image dataset
	@author: JungiLee
"""
import os
import torch
import torchvision
import numpy as np

from dataloader.Image import FolderDataset
from dataloader.BaseDataModule import BaseDataModule

class Tinyimagenet(BaseDataModule):
	""" Tinyimagenet module 
		Args:
			- root (str): data directory
			- batch_size (int): the number of batch
			- num_workers (int): the number of workers
			- seed (int): seed for reproducibility
			- train_transform (func): Transform for train data
			- test_transform (func): Transform for test data
	"""
	def __init__(
					self,
					root="/workspace/code/data",
					batch_size=256,
					num_workers=4,
					seed=0,
					train_transform=None,
					test_transform=None
				):
		super().__init__(
							root=root,
							batch_size=batch_size,
							num_workers=num_workers,
							seed=seed,
							train_transform=train_transform,
							test_transform=test_transform
						)

	def _setup(self):
		#Trian&Test data setup
		train_gt = []
		train_data=[]
		label_dict = {}
		train_path = os.path.join(self.root,'tiny-imagenet-200/train/')
		n_class = 0
		for fold in os.listdir(train_path):
			label_dict[fold]=n_class
			class_path=os.path.join(train_path,fold,'images')
			for fname in os.listdir(class_path):
				train_data.append(os.path.join(class_path,fname))
				train_gt.append(n_class)
			n_class +=1

		test_gt = []
		test_data=[]
		test_path = os.path.join(self.root,'tiny-imagenet-200/val')
		for i, line in enumerate(open(os.path.join(test_path,'val_annotations.txt'), 'r')):
			a = line.split('\t')
			img, cls_id = a[0], a[1]
			test_data.append(os.path.join(test_path,'images',img))
			test_gt.append(label_dict[cls_id])


		self.train_data = FolderDataset(train_data, train_gt, self.train_transform)
		self.test_data = FolderDataset(test_data, test_gt, self.test_transform)


