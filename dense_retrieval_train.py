from retrieval_module.utills import *
from retrieval_module.retrieval_dataset import *

from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaModel
from transformers import HfArgumentParser, TrainingArguments

import torch
import torch.nn.functional as F

from arguments import (
    DataTrainingArguments,
    CustomArguments
)

from tqdm import trange, tqdm
from utills.utills import config_setting_for_dense_retrieval
import wandb

def train(tokenizer, q_encoder, p_encoder, optimizer, scheduler, train_dataloader, valid_context, valid_question, data_args):
    print('Start train!!')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  
    q_encoder.zero_grad()
    p_encoder.zero_grad()
    torch.cuda.empty_cache()
    best_metric_top_1 = 0
    best_metric_top_3 = 0
    best_metric_top_10 = 0
    best_metric_top_35 = 0

    train_iterator = trange(int(data_args.dense_train_epoch), desc="Epoch")
    for epoch in train_iterator:    
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")    
        # Train
        train_loss = train_per_epoch(q_encoder, p_encoder, optimizer, epoch_iterator, data_args)
        
        # Valid
        top_1_acc, top_3_acc, top_10_acc, top_35_acc, top_100_acc = valid_per_epoch(tokenizer, p_encoder, q_encoder, valid_context, valid_question, data_args)

        # logging
        print(f'epoch: {epoch} | train_loss:{train_loss:.5f} | '
              f'top-1 acc: {top_1_acc:.2f} | '
              f'top-3 acc: {top_3_acc:.2f} | '
              f'top-10 acc: {top_10_acc:.2f} | '
              f'top-35 acc: {top_35_acc:.2f} | '
              f'top-100 acc: {top_100_acc:.2f} | ')
		
        scheduler.step()

        # 에폭 단위 저장       
        q_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/{epoch}ep/q_encoder')
        p_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/{epoch}ep/p_encoder')
        print(f'{epoch} saved!')

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'top-1 acc': top_1_acc,
            'top-3 acc': top_3_acc,
            'top-10 acc': top_10_acc,
            'top-35 acc': top_35_acc,
            'top-100 acc': top_100_acc,
        })

        # best 모델 저장 top_1_acc 기준
        if top_1_acc > best_metric_top_1:
            best_metric_top_1 = top_1_acc
            q_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/best/q_encoder')
            p_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/best/p_encoder')
            print('best top-1 saved!')
        elif top_3_acc > best_metric_top_3:
            best_metric_top_3 = top_3_acc
            q_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/best/q_encoder')
            p_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/best/p_encoder')
            print('best top-3 saved!')
        elif top_10_acc > best_metric_top_10:
            best_metric_top_10 = top_10_acc
            q_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/best/q_encoder')
            p_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/best/p_encoder')
            print('best top-10 saved!')
        elif top_35_acc > best_metric_top_35:
            best_metric_top_35 = top_35_acc
            q_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/best/q_encoder')
            p_encoder.save_pretrained(f'{data_args.dense_train_output_dir}/best/p_encoder')
            print('best top-35 saved!')

def train_per_epoch(q_encoder, p_encoder, optimizer, epoch_iterator, data_args):
    batch_loss = 0
    for step, batch in enumerate(epoch_iterator):
      q_encoder.train()
      p_encoder.train()
      
      if torch.cuda.is_available():
        batch = tuple(t.cuda() for t in batch)

      p_inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]
                  }
      
      q_inputs = {'input_ids': batch[3],
                  'attention_mask': batch[4],
                  'token_type_ids': batch[5]}
      
      p_outputs = p_encoder(**p_inputs).pooler_output  # (batch_size, emb_dim)
      q_outputs = q_encoder(**q_inputs).pooler_output  # (batch_size, emb_dim)

      # Calculate similarity score & loss
      sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

      # target: position of positive samples = diagonal element 
      targets = torch.arange(0, len(batch[0])).long()
      if torch.cuda.is_available():
        targets = targets.to('cuda')

      sim_scores = F.log_softmax(sim_scores, dim=1)

      loss = F.nll_loss(sim_scores, targets)

      loss.backward()
      optimizer.step()
      q_encoder.zero_grad()
      p_encoder.zero_grad()

      batch_loss += loss.detach().cpu().numpy()
      torch.cuda.empty_cache()
    return batch_loss / len(epoch_iterator)

def valid_per_epoch(tokenizer, p_encoder, q_encoder, valid_context, valid_question, data_args):
    print(f'Valid start!')
    with torch.no_grad():
        p_encoder.eval()

        p_embs = []
        for p in valid_context:
            p = tokenizer(p, max_length=data_args.dense_max_length, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            p_emb = p_encoder(**p).pooler_output.to('cpu').numpy()
            p_embs.append(p_emb)

        p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)

    top_1 = 0
    top_3 = 0
    top_10 = 0
    top_25 = 0
    top_35 = 0
    top_100 = 0
    q_encoder.eval()
    for sample_idx in tqdm(range(len(valid_question))):
        query = valid_question[sample_idx]

        q_seqs_val = tokenizer([query], max_length=80, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        q_emb = q_encoder(**q_seqs_val).pooler_output.to('cpu')  #(num_query, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        if sample_idx == rank[0]: 
            top_1 += 1
        if sample_idx in rank[0:3]: 
            top_3 += 1
        if sample_idx in rank[0:10]: 
            top_10 += 1
        if sample_idx in rank[0:25]: 
            top_25 += 1
        if sample_idx in rank[0:35]: 
            top_35 += 1
        if sample_idx in rank[0:100]: 
            top_100 += 1
    
    return top_1/len(valid_question) * 100, top_3/len(valid_question) * 100, top_10/len(valid_question) * 100, top_35/len(valid_question) * 100, top_100/len(valid_question) * 100

def main():
    parser = HfArgumentParser(
        (DataTrainingArguments, CustomArguments)
    )
    data_args, cus_args = parser.parse_args_into_dataclasses()

    if cus_args.use_wandb:
        config = config_setting_for_dense_retrieval(data_args, cus_args)
        wandb.init(project=cus_args.project_name, entity=cus_args.entity_name, name=cus_args.wandb_run_name, config=config)


    # tokenizer 준비
    print('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(data_args.dense_base_model)

    # 학습 및 검증 데이터 준비
    print('Loading data')
    train_dataloader, valid_context, valid_question = prepare_data(tokenizer, data_args.dense_train_batch_size, data_args.dense_max_length)
 
    # 모델 준비
    print('Loading models')
    p_encoder = RobertaModel.from_pretrained(data_args.dense_base_model)
    q_encoder = RobertaModel.from_pretrained(data_args.dense_base_model)
    
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=data_args.dense_train_learning_rate, eps=1e-08, weight_decay=0.01)
    t_total = len(train_dataloader) // 1 * 50 #(gradient_accumulation_steps, epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=t_total)
    
    if cus_args.use_wandb:
        wandb.watch(p_encoder)

    # 훈련 시작!
    train(tokenizer, q_encoder, p_encoder, optimizer, scheduler, train_dataloader, valid_context, valid_question, data_args)

if __name__=='__main__':
    main()