"""
	DSBA Pretraining Code
	Copyright 2025. Jungi Lee All rights reserved.

	@description: logger
	@author: JungiLee
"""


import os
import logging
import datetime
import shutil
import matplotlib.pyplot as plt
import numpy as np


def log(string):
	""" 
		Print string in console and write string in log file(logger and backup logger).
		Paramters
			- string (str): print string
	"""
	logger = logging.getLogger(f"logger")
	logger.info(f"{tgt_device} - {string}")

def log_img(image, fname):
	plt.clf()
	plt.imshow(image)
	plt.savefig(os.path.join(ckpt_path, f"{fname}.pdf"), dpi=1200)
	plt.savefig(os.path.join(backup_path, f"{fname}.pdf"), dpi=1200)

def vanillagradient(model, image, label, device):
	"""
		Visualization via Vanilla Gradient Map
		Parameters
			model (nn.Module): torch model
			image (ndarray): input image
			device (torch): torch model device
		Returns:
			visualization image save
	"""

	model = model.to(device)
	model.eval()

	input_tensor = image.unsqueeze(0).to(device)
	input_tensor.requires_grad_()

	output = model(input_tensor)
	pred_class = output.argmax()

	model.zero_grad()
	output[0, pred_class].backward()

	gradients = input_tensor.grad.data.squeeze().cpu().numpy()  # shape: [3, H, W]
	saliency = np.max(np.abs(gradients), axis=0)  # [H, W]


	log_img(image.permute(1,2,0).cpu().numpy(), fname=f"Origin{label}")
	log_img(saliency, fname=f"VanillaGradient{label}")




	
def make_logger(model_name,device, ckpt_dir, log_dir):
	""" 
		Make logger and backup logger. Logger is INFO-level logger and backup logger is DEBUG-level logger. Logge is saved in "./ckpt/info.log" and backup logger is saved in "./log/Y-M-D-H-M/info.log" format.
		Parameters
			- model_name (str): model name
			- device (str): model device
			- ckpt_dir(str): checkpoint directory path
			- backup_path (str): backup (log) directory path
	"""
	global tgt_device
	global ckpt_path
	global backup_path 

	#File exist Check and generate directory
	if not os.path.isdir(ckpt_dir):
		os.mkdir(ckpt_dir)
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)

	#Set the limit of the number of log files
	log_file = [s for s in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, s))]
	log_max = 100
	if len(log_file) > log_max:
		log_file.sort(key=lambda s: os.path.getmtime(os.path.join(log_dir,s)), reverse=True)
		for i, file_name in enumerate(log_file):
			if i >= log_max:
				shutil.rmtree(f"{log_dir}/{file_name}")

	#Set Checkpoint path
	ckpt_path=f"{ckpt_dir}/{model_name}_{device}"
	if not os.path.isdir(ckpt_path):
		os.mkdir(ckpt_path)

	#Set backup path
	backup_path = log_dir + "/"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") 
	if not os.path.isdir(backup_path):
		os.mkdir(backup_path)


	tgt_device=device
	#Make logger
	logger = logging.getLogger(f"logger")
	if len(logger.handlers) > 0:
		logger.handlers.clear() 
		logger = logging.getLogger()
		logger.handlers.clear()
		logger = logging.getLogger(f"logger")

	logging.basicConfig(filename=ckpt_path+"/info.log",level=logging.INFO, filemode='w')
	logger.setLevel(logging.INFO)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler = logging.FileHandler(backup_path+"/info.log")
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	stream_handler= logging.StreamHandler()
	stream_handler.setLevel(logging.INFO)
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
