import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import BertPreTrainedModel, BertModel, RobertaModel, RobertaConfig, RobertaPreTrainedModel

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

class RobertaEncoder(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        config = RobertaConfig().from_pretrained(model_name)
        self.encoder = RobertaModel(config).from_pretrained(model_name)
        self.dense = nn.Linear(768, 768, bias=True)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = nn.Tanh()(x.pooler_output)
        x = self.dense(x)
        x = nn.Tanh()(x)
        return x

class BertEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = RobertaModel(config)
        #self.init_weights()

    def forward(self, input_ids, 
              attention_mask=None, token_type_ids=None): 

        outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)

        return outputs

    