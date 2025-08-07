"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: CIFAR100 image dataset
	@author: JungiLee
"""

import torch
import torchvision
import numpy as np

from dataloader.Image import ImageDataset
from dataloader.BaseDataModule import BaseDataModule

class CIFAR100(BaseDataModule):
	""" CIFAR100 module 
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
		train_data = torchvision.datasets.CIFAR100(root=self.root, train=True, download=True)
		test_data = torchvision.datasets.CIFAR100(root=self.root, train=False, download=True)

		train_data, train_gt = train_data.data, train_data.targets
		test_data, test_gt = test_data.data, test_data.targets

		train_data, train_gt = torch.as_tensor(train_data), torch.as_tensor(train_gt)
		test_data, test_gt = torch.as_tensor(test_data), torch.as_tensor(test_gt)

		train_data, test_data = train_data.permute(0,3,1,2), test_data.permute(0,3,1,2)

		self.train_data = ImageDataset(train_data, train_gt, self.train_transform)
		self.test_data = ImageDataset(test_data, test_gt, self.test_transform)


