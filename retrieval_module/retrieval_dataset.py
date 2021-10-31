import re
import torch

from transformers import BertPreTrainedModel, BertModel
from torch.utils.data import Dataset


class RetrievalTrainDataset(Dataset):
    def __init__(self, input_ids, attention_mask, q_input_ids, q_attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.q_input_ids = q_input_ids
        self.q_attention_mask = q_attention_mask
        
    def __getitem__(self, idx):
        #return self.input_ids[idx], self.attention_mask[idx], self.q_input_ids[idx], self.q_attention_mask[idx]
        item = {
             'input_ids': self.input_ids[idx],
             'attention_mask': self.attention_mask[idx]
        }
        item2 = {
            'input_ids': self.q_input_ids[idx],
            'attention_mask': self.q_attention_mask[idx]
        }
        return item, item2
        
    def __len__(self):
        return len(self.input_ids)

class RetrievalValidDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.input_ids = input_ids.to(self.device)
        self.attention_mask = attention_mask.to(self.device)
        
    def __getitem__(self, idx):
        item = {
             'input_ids': self.input_ids[idx],
             'attention_mask': self.attention_mask[idx]
        }        
        return item
        
    def __len__(self):
        return len(self.input_ids)

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, 
              attention_mask=None, token_type_ids=None): 

        outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)

        return outputs

    