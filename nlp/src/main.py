import wandb 
from tqdm import tqdm
import os
import wandb
import torch
import torch.nn as nn

from model import EncoderForClassification
from data import get_dataloader

# torch.cuda.set_per_process_memory_fraction(11/24) -> 김재희 로컬과 신입생 로컬의 vram 맞추기 용도. 과제 수행 시 삭제하셔도 됩니다. 
# model과 data에서 정의된 custom class 및 function을 import합니다.
"""
여기서 import 하시면 됩니다. 
"""
import numpy as np
from utils import parse
from transformers import set_seed
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def train_iter(model, batch, optimizer, criterion, device):
	X= {k: v.to(device) for k, v in batch[0].items()}
	Y=batch[1].to(device)
	prob = model(X)
	loss = criterion(prob,Y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	wandb.log({'train_loss': loss.item()})
	return loss.item()



def valid_iter(model, loader, criterion, device):
	n=0
	acc=0
	loss_total = []
	model.eval()
	with torch.no_grad():
		for batch in loader:
			X= {key : value.to(device) for key, value in batch[0].items()}
			Y=batch[1].to(device)
			prob= model(X)
			loss=criterion(prob,Y)
			loss_total.append(loss.item())
			n+=len(Y)
			acc+=calculate_accuracy(prob, Y)	 
	return np.mean(loss_total), acc/n

def calculate_accuracy(logits, label):
	preds = logits.argmax(dim=-1)
	correct = (preds == label).sum().item()
	return correct 

def main(cfg) :
	wandb.init(project="DSBA pretraining", name=f"{cfg['model']}", config=cfg)

	# Set device
	device= cfg['device']
	set_seed(cfg['seed'])
	if cfg['model'] == 'bert':
		model_id = "bert-base-uncased"
	elif cfg['model'] == 'modernbert':
		model_id = "ModernBERT-base"
	else:
		raise ValueError(f"Model name ({cfg['model']}) is not available")


	if os.path.exists(cfg['ckpt_path'])==False:
		os.mkdir(cfg['ckpt_path'])

	# Load model
	model = EncoderForClassification(cfg, model_id).to(device)

	# Load data
	data = load_dataset("imdb")
	total_data = np.hstack((data['train']['text'], data['test']['text']))
	total_label= np.hstack((data['train']['label'], data['test']['label']))
	train_x, test_x, train_y, test_y = train_test_split(total_data, total_label, test_size=0.2, random_state=cfg['seed'], stratify=total_label)
	valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=cfg['seed'], stratify=test_y)

	train_loader = get_dataloader(cfg, train_x, train_y, model_id, split="train")	
	valid_loader = get_dataloader(cfg, valid_x, valid_y, model_id, split="valid")	
	test_loader = get_dataloader(cfg, test_x, test_y, model_id, split="test")	

	# Set optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
	criterion = nn.CrossEntropyLoss().to(device)


	# Train & validation for each epoch
	total_batch = len(train_x)
	valid_loss=[]
	for epoch in tqdm(range(1,cfg['epoch']+1)):
		#Train model
		model.train()
		train_loss= []
		for batch in train_loader:
			loss = train_iter(model, batch, optimizer, criterion, device)
			train_loss.append(loss)
		wandb.log({'train_avg_loss': np.mean(train_loss)}, step=epoch)

		#Validate model
		loss, acc = valid_iter(model, valid_loader, criterion, device)
		valid_loss.append(loss)
		wandb.log({'valid_avg_loss': loss}, step=epoch)
		wandb.log({'valid_acc': acc}, step=epoch)

		#model save
		torch.save(model.state_dict(), os.path.join(cfg['ckpt_path'], f"{cfg['model']}_{epoch-1}.pth"))

	# Test last model
	loss, acc = valid_iter(model, test_loader, criterion, device)
	wandb.log({'Test Accuracy of last model': acc})
	print(f"Test Accuracy (last model): {acc:.3f}")

	# Test best model
	idx = np.argmin(valid_loss)
	model.load_state_dict(torch.load(os.path.join(cfg['ckpt_path'], f"{cfg['model']}_{idx}.pth")))
	loss, acc = valid_iter(model, test_loader, criterion, device)
	wandb.log({'Test Accuracy of bestmodel': acc})
	print(f"Test Accuracy (best model): {acc:.3f}")

	
if __name__ == "__main__" :
	parser = parse()
	cfg = parser.parse_args()
	cfg = vars(cfg)
	main(cfg)
