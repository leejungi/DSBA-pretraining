from transformers import AutoTokenizer
from datasets import load_dataset

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Union, List, Tuple, Literal
from transformers import AutoTokenizer
import numpy as np
import copy

def to_tensor(x):
	if type(x)==list:
		x = np.array(x)
	return torch.from_numpy(x)


class IMDBDatset(torch.utils.data.Dataset):
	def __init__(self, cfg, data, label, model_id):
		self.cfg=cfg
		self.data = data
		self.label = label
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
	   
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx) -> Tuple[dict, int]:
		data = copy.deepcopy(self.data[idx])
		data = self.tokenizer(data, return_tensors="pt", padding="max_length", truncation=True, max_length=self.cfg['max_length'])
		data['label']=copy.deepcopy(self.label[idx])
		return data



	@staticmethod
	def collate_fn(batch : List[Tuple[dict, int]]) -> dict:
		"""
		Inputs :
			batch : List[Tuple[dict, int]]
		Outputs :
			data_dict : dict{
				input_ids : torch.Tensor
				token_type_ids : torch.Tensor
				attention_mask : torch.Tensor
				label : torch.Tensor
			}
		"""
		labels = to_tensor([batch[i]['label'] for i in range(len(batch))])
		outputs={}
		for k in batch[0].keys():
			if k=="label":
				continue
			outputs[k] = to_tensor([batch[i][k] for i in range(len(batch))]).squeeze(1)
		return outputs, labels
	
def get_dataloader(cfg, data, label, model_id, split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
	"""
	Output : torch.utils.data.DataLoader
	"""
	dataset = IMDBDatset(cfg, data, label, model_id)
	dataloader = DataLoader(dataset, batch_size=cfg['batch'], shuffle=(split=='train'), collate_fn=IMDBDatset.collate_fn, num_workers=4)
	return dataloader
