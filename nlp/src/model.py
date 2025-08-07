from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from transformers import AutoModel

class EncoderForClassification(nn.Module):
	def __init__(self, cfg, model_id):
		super().__init__()
		self.encoder = AutoModel.from_pretrained(model_id)
		self.cls = nn.Linear(768,2)
	
	def forward(self, x):

		inputs = self.encoder(**x)
		cls_token = inputs.last_hidden_state[:,0,:]
		return self.cls(cls_token)
