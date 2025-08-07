"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: Printer
	@author: JungiLee
"""

import os
from termcolor import cprint

def status(state:str, current_epoch:int, total_epoch:int, current_batch:int, total_batch:int, current_loss:float, loss_mean:float):
	"""
		status update printer
		Parameters
			- state (str): the prefix of string
			- current_epoch (int): current epoch
			- total_epoch (int): total epoch
			- current_batch (int): current batch
			- total_batch (int): total batch
			- current loss (float): loss in current batch
			- loss_mean (float): one epoch average of losses
	"""
	if current_epoch == None or total_epoch == None:
		msg = "[{}]-[Batch {}/{}] [Current Loss: {:.5f}(Avg: {:.5f})]".format(
			state,
			current_batch, total_batch,
			current_loss, loss_mean
		)
		color = "yellow"
	else:
		msg = "[{}]-[Epoch {}/{}] [Batch {}/{}] [Current Loss: {:.5f}(Avg: {:.5f})]".format(
			state,
			current_epoch, total_epoch,
			current_batch, total_batch,
			current_loss, loss_mean
		)
		color = "cyan"
	cprint(msg, color=color, end="\r")
