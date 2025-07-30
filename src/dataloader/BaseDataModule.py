"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: Baseline data module
	@author: JungiLee
"""


import numpy as np
import torch
import random
from torch.utils.data.dataloader import DataLoader
from abc import ABC, abstractmethod
from collections import Counter



class BaseDataModule(ABC):
	""" Baseline data module 
		Args:
			- root (str): data directory
			- batch_size (int): the number of batch
			- num_workers (int): the number of workers
			- seed (int): seed for reproducibility
			- train_transform (func): Transform for train data
			- test_transform (func): Transform for test data

		Attributes:
			- is_setup (bool): flag for setup
	"""
	def __init__(
					self,
					root,
					batch_size=256,
					num_workers=4,
					seed=0,
					train_transform=None,
					test_transform=None,
				):
		self.root=root
		self.batch_size=batch_size
		self.num_workers=num_workers
		self.seed=seed
		self.train_transform=train_transform
		self.test_transform=test_transform

		self.valid_flag= False
		self.is_setup = False

	@abstractmethod
	def _setup(self):
		raise NotImplementedError

	def setup(self):
		""" Prepare dataset """
		self._setup()
		self.is_setup = True
		return self

	def seed_worker(self, worker_id):
		""" Fix seed for reproducibility """
		worker_seed = torch.initial_seed() % 2**32
		np.random.seed(worker_seed)
		random.seed(worker_seed)

	def generator(self):
		""" For reproducibility """
		g = torch.Generator()
		g.manual_seed(self.seed)
		return g

	def train_dataloader(self):
		""" Return train data loader """
		train_batch = min(len(self.train_data), self.batch_size)

		return DataLoader(
							self.train_data, 
							shuffle=True,
							drop_last=True,
							batch_size=train_batch,
							num_workers=self.num_workers,
							persistent_workers=True if self.num_workers >0 else False,
							worker_init_fn=self.seed_worker,
							generator=self.generator()
						)

	def test_dataloader(self):
		""" Return test data loader """
		test_batch = min(len(self.test_data), self.batch_size)
		return DataLoader(
							self.test_data, 
							shuffle=False,
							drop_last=False,
							batch_size=test_batch,
							num_workers=self.num_workers,
							persistent_workers=True if self.num_workers >0 else False,
							worker_init_fn=self.seed_worker,
							generator=self.generator()
						)

	def get_traindata(self):
		""" Return train data """
		return self.train_data.get_data()

	def get_testdata(self):
		""" Return test data """
		return self.test_data.get_data()

	def get_traingt(self):
		""" Return train gt """
		return self.train_data.get_gt()

	def get_testgt(self):
		""" Return test gt """
		return self.test_data.get_gt()

	def set_traintransform(self, transform):
		""" Set train transform"""
		return self.train_data.set_transform(transform)

	def set_testtransform(self, transform):
		""" Set test transform"""
		return self.test_data.set_transform(transform)
