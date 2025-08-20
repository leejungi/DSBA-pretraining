from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Union, List, Tuple, Literal
from transformers import set_seed
from collections import Counter

set_seed(42)

class IMDBDatset(torch.utils.data.Dataset):
	def __init__(self, data_config, split : Literal['train', 'valid', 'test']):
		self.split = split
		model_id = "answerdotai/ModernBERT-base" #"ModernBERT-base"
		self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
		self.max_len = 128
		self.valid_size = 0.1
		self.test_size = 0.1

		# data loading
		imdb = load_dataset('imdb')
		combined_dataset = concatenate_datasets([imdb['train'], imdb['test']])
		trainval_test = combined_dataset.train_test_split(test_size=self.test_size, seed=42)
		trainval = trainval_test['train']
		test = trainval_test['test']
		train_valid = trainval.train_test_split(test_size=self.valid_size, seed=42)
		train = train_valid['train']
		valid = train_valid['test']
		if self.split == 'train':
			data = train
		elif self.split == 'valid':
			data = valid
		elif self.split == 'test':
			data = test
		tokenized_data = data.map(lambda example: self.tokenizer(example['text'], truncation=True, padding='max_length', max_length=self.max_len))
		self.data = tokenized_data.to_dict()
		print(Counter(self.data['label']))

	def __getitem__(self, idx) -> Tuple[dict, int]:
		input_data = {
			"input_ids": self.data['input_ids'][idx],
			"attention_mask": self.data['attention_mask'][idx],
			"label": self.data['label'][idx]
		}
		if 'token_type_ids' in self.data:
			input_data["token_type_ids"] = self.data['token_type_ids'][idx]
		return input_data
			
	def __len__(self):
		return len(self.data['input_ids'])

	@staticmethod
	def collate_fn(batch: List[Tuple[dict, int]]) -> dict:
		data_dict = {"input_ids": [], "attention_mask": [], "label": []}

		if 'token_type_ids' in batch[0]:
			data_dict["token_type_ids"] = []

		for data in batch:
			data_dict["input_ids"].append(data["input_ids"])
			data_dict["attention_mask"].append(data["attention_mask"])
			data_dict["label"].append(data["label"])
			if "token_type_ids" in data:
				data_dict["token_type_ids"].append(data["token_type_ids"])

		data_dict["input_ids"] = torch.tensor(data_dict["input_ids"])
		data_dict["attention_mask"] = torch.tensor(data_dict["attention_mask"])
		data_dict["label"] = torch.tensor(data_dict["label"])

		if "token_type_ids" in data_dict:
			data_dict["token_type_ids"] = torch.tensor(data_dict["token_type_ids"])

		return data_dict
	
def get_dataloader(data_config, split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
	dataset = IMDBDatset(data_config, split)
	dataloader = DataLoader(dataset,
							batch_size=data_config['batch'],
							shuffle=(split=='train'), # shuffle train only
							collate_fn=IMDBDatset.collate_fn)
	return dataloader
