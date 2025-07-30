"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: dataset for Image dataset
	@author: JungiLee
"""
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode

class ImageDataset(torch.utils.data.Dataset):
	"""
		Image dataset for model
		Args:
			- data(list): merged data that is used for train, valid, and test
			- gt(list): ground truth
			- transform (func): data transform
		Attributes:
			- length (int): the number of data
	"""

	def __init__(self, data, gt, transform=None):
		self._data = data
		self._gt = gt
		self.transform = transform
		self.len = np.shape(self._data)[0]

	def __len__(self):
		"""
			return length attributes
		"""
		return self.len

	def __getitem__(self, idx):
		"""
			Args	
				- idx (int): the idx ranged from 0 to length
			Returns
				- data (torch): the image indexed by input
				- gt (torch): the label of idx
		"""
		data = self._data[idx]
		gt = self._gt[idx]

		if self.transform:
			if data.ndim==3:
				data = transforms.ToPILImage()(data)
			data= self.transform(data)
		data = data.float()
		return data, gt

	def get_data(self):
		return self._data

	def get_gt(self):
		return self._gt

	def set_transform(self,transform):
		self.transform=transform
		return self


class FolderDataset(torch.utils.data.Dataset):
	"""
		Folder dataset for model
		Args:
			- data(list): merged data that is used for train, valid, and test
			- gt(list): ground truth
			- transform (func): data transform
		Attributes:
			- length (int): the number of data
	"""

	def __init__(self, data, gt, transform=None):
		self._data = data
		self._gt = gt
		self.transform = transform
		self.len = np.shape(self._data)[0]

	def __len__(self):
		"""
			return length attributes
		"""
		return self.len

	def __getitem__(self, idx):
		"""
			Args	
				- idx (int): the idx ranged from 0 to length
			Returns
				- data (torch): the image indexed by input
				- gt (torch): the label of idx
		"""
		data_path = self._data[idx]
		data = read_image(data_path)
		if data.shape[0] == 1:
			data = read_image(data_path,ImageReadMode.RGB)
		gt = self._gt[idx]

		if self.transform:
			if data.ndim==3:
				data = transforms.ToPILImage()(data)
			data= self.transform(data)
		data = data.float()
		return data, gt

	def get_data(self):
		return self._data

	def get_gt(self):
		return self._gt

	def set_transform(self,transform):
		self.transform=transform
		return self
