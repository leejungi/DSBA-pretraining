"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: Classifier Module
	@author: JungiLee
"""

import time
import torch
import numpy as np
import torch.nn as nn
import timm
from torchinfo import summary
from tqdm import tqdm
#import wandb

from utils.log import log, log_img, vanillagradient
from utils.printer import status
from network import resnet

n_cls = {
			"cifar10": 10,
			"cifar100": 100,
			"tinyimagenet": 200
		}

class CLSModule:
	def __init__(self, cfg):
		"""
			CLSModule initializer
			Parameters
				- cfg (dict): parsing argument
				- model (nn.Module): Deep learning model
		"""
		self.cfg = cfg
		self.device = cfg['device']
		self.epoch= cfg['epoch']
		self.lr= cfg['lr']
		self.wd= cfg['wd']

		#Build model
		self.model = self.build_model(self.cfg['model'], n_class=n_cls[self.cfg['dataset'].lower()], pretrained=self.cfg['pretrained'])

		# Loss function
		self.criterion = nn.CrossEntropyLoss().to(self.device)

		# Optimizer 
		if self.cfg['optimizer']=="Adam":
			self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
		elif self.cfg['optimizer']=="SGD":
			self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=0.9)
		else:
			raise ValueError(f"{optimizer} is undefined optimizer")

		# Scheduler Selection
		if self.cfg['scheduler']=="cosine":
			self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optim, T_max=self.epoch)
		elif self.cfg['scheduler']==None or self.cfg['scheduler']=="":
			self.scheduler = None
		else:
			raise ValueError(f"{self.cfg['scheduler']} Unknown scheduler name")


	def build_model(self, model_name, n_class, pretrained=False):
		"""
			Build model
			Parameters
				- model_name (str): model name
				- n_class (int): the number of classes
				- pretrianed (bool): boolean for pretrained model
			Returns
				- model (torch): torch model
		"""
		pretrained_flag = True if pretrained >0 else False

		if 'vit' in model_name.lower():
			model=timm.create_model('vit_small_patch16_224', pretrained=pretrained_flag, num_classes=n_class)
			if pretrained==2:
				for param in model.parameters():
					param.requires_grad = False
				for param in model.head.parameters():
					param.requires_grad = True
		else:
			if model_name =="R18":
				model = resnet.resnet18(pretrained=pretrained_flag)
			elif model_name =="R34":
				model = resnet.resnet34(pretrained=pretrained_flag)
			elif model_name =="R50":
				model = resnet.resnet50(pretrained=pretrained_flag)
				model.in_layer=2048

			if pretrained==0 and n_class<=100:
				model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)
				model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0) # Identity
			elif pretrained==2:
				for param in model.parameters():
					param.requires_grad = False

			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs, n_class)

		log(summary(model, (2,3,224,224), verbose=0))
		model.to(self.device)


		return model

	def save(self, path):
		"""
			Model save function
			Parameters
				- path (str): save path
		"""
		torch.save(self.model.state_dict(), path)

	def load(self, path):
		"""
			Model load function
			Parameters
				- path (str): load path
		"""
		self.path = path
		self.model.load_state_dict(torch.load(path))


	def fit(self, loader):
		"""
			Train function
			Parameters
				- loader (torch.utils.data.DataLoader): data loader 
		"""
		self.model.train()
		loss_total = []
		total_batch=len(loader)
		for epoch in tqdm(range(self.epoch)):
			#Training
			for batch_idx, batch in enumerate(loader):
				X =batch[0].to(self.device)
				Y =batch[1].to(self.device)

				prob = self.model(X)
				loss = self.criterion(prob, Y)

				self.optim.zero_grad()
				loss.backward()
				self.optim.step()
				loss_total.append(loss.item())
				status(f"Train", epoch, self.epoch, batch_idx+1, total_batch, loss.item(), np.mean(loss_total))
#			wandb.log({'train_loss': np.mean(loss_total)}, step=epoch)

			if self.scheduler != None:
				self.scheduler.step()
		return self


	def predict(self, loader):
		"""
			prediction step. 
			Parameters
				- loader (torch.utils.data.DataLoader): data loader for validation and test data
			Returns
				- result_dict (dict): information (metric) dictionary
		"""
		loss_total = []
		pred_list = []
		label_list = []
		time_list = []
		total_batch=len(loader)
		with torch.no_grad():
			self.model.eval()
			for batch_idx, batch in enumerate(loader):
				batch_start = time.time()
				X = batch[0].to(self.device)
				Y = batch[1].to(self.device)

				prob = self.model(X)
				loss = self.criterion(prob, Y)
				loss_total.append(loss.item())
				pred= torch.argmax(prob, 1)

				time_list.append(time.time()-batch_start)

				pred_list += list(pred.to('cpu').numpy())
				label_list += list(Y.to('cpu').numpy())
				status("Test", None, None, batch_idx+1, total_batch, loss.item(), np.mean(loss_total))


			batch_time = np.mean(time_list[:-1])
			pred_list= np.array(pred_list)
			label_list= np.array(label_list)
			acc = (pred_list==label_list).mean()
			
			result_dict={
							'accuracy': acc,
							'inference':batch_time,
							'loss_total': np.sum(loss_total),
							'loss_mean': np.mean(loss_total),
						}
			log(f"Visualization class {Y[0]}")
		vanillagradient(self.model, X[0], Y[0], device=self.device)

		return result_dict



