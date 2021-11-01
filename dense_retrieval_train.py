from retrieval_module.utills import *
from retrieval_module.retrieval_dataset import *

from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaModel, BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaEncoder

import torch
from torch.nn import TripletMarginLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F


def main():
    num_negative = 11
    max_seq_length = 384
    batch_size = 1
    triplet_loss = TripletMarginLoss(margin=1.5, p=2)
    # tokenizer 준비
    model_name = 'klue/roberta-small' # "bert-base-multilingual-cased"#
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 학습 및 검증 데이터 준비
    train_dataloader, valid_loader, valid_q_loader, ground_truth = prepare_data(tokenizer, max_seq_length, num_negative)
 
    # 모델 준비
    #p_encoder = BertEncoder.from_pretrained(model_name)
    #q_encoder = BertEncoder.from_pretrained(model_name)
    p_encoder = BertEncoder.from_pretrained('kiyoung2/dpr_p-encoder_roberta-small')
    q_encoder = BertEncoder.from_pretrained('kiyoung2/dpr_q-encoder_roberta-small')
    # p_encoder = RobertaEncoder.from_pretrained(model_name)
    # q_encoder = RobertaEncoder.from_pretrained(model_name)

    #RobertaEncoder
    # p_encoder = RobertaEncoder(model_name)
    # q_encoder = RobertaEncoder(model_name) 
    
    # for param in p_encoder.parameters():
    #     param.requires_grad = False

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # p_encoder.parameters(),
        # q_encoder.parameters()
        {'params': [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-1, eps=1e-08, weight_decay=0.01)
    t_total = len(train_dataloader) // 1 * 50 #(gradient_accumulation_steps, epoch)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=t_total)
    scheduler = CosineAnnealingLR(optimizer, 10)
    # 훈련 시작!
    train(tokenizer, q_encoder, p_encoder, optimizer, scheduler, triplet_loss,  train_dataloader, valid_loader, valid_q_loader, ground_truth, batch_size, num_negative)

def train(tokenizer, q_encoder, p_encoder, optimizer, scheduler, criterion, train_dataloader, valid_loader, valid_q_loader, ground_truth, batch_size, num_negative):
    print('Start train!!')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  
    q_encoder.zero_grad()
    p_encoder.zero_grad()
    torch.cuda.empty_cache()
    best_metric = 0

    for epoch in range(0, 51):        
        # Train
        train_loss = train_per_epoch(tokenizer, q_encoder, p_encoder, optimizer, criterion, train_dataloader, batch_size, num_negative, device)
        
        # Valid
        top_1_acc, top_3_acc, top_5_acc = valid_per_epoch(p_encoder, q_encoder, valid_loader, valid_q_loader, ground_truth)

        # logging
        print(f'epoch: {epoch} | train_loss:{train_loss:.5f} top-1 acc: {top_1_acc*100:.2f} | top-3 acc: {top_3_acc*100:.2f} | top-5 acc: {top_5_acc*100:.2f}')
		
        scheduler.step()
        # 10 에폭 단위 저장
        if epoch % 10 == 0 and epoch > 0:
            q_encoder.save_pretrained(f'./models_result/roberta-base_retriever/{epoch}ep/q_encoder')
            p_encoder.save_pretrained(f'./models_result/roberta-base_retriever/{epoch}ep/p_encoder')
            print('{epoch} saved!')
        
        # best 모델 저장 top_1_acc 기준

        if top_1_acc > best_metric:
            best_metric = top_1_acc
            q_encoder.save_pretrained(f'./models_result/roberta-base_retriever/best/q_encoder')
            p_encoder.save_pretrained(f'./models_result/roberta-base_retriever/best/p_encoder')
            print('best saved!')

def train_per_epoch(tokenizer, q_encoder, p_encoder, optimizer, criterion, train_dataloader, batch_size, num_negative, device):
    batch_loss = 0
    optimizer.zero_grad()

    for step, (passage_item, question_item) in enumerate(tqdm(train_dataloader)):
        q_encoder.train() # (batch_size, 4, 384) 
        #p_encoder.train()

        targets = torch.zeros(batch_size).long().to(device)
        p_inputs = {'input_ids': passage_item['input_ids'].view(
                                    batch_size*(num_negative), -1).to(device), # (batch_size * 4, 384)  =
                'attention_mask': passage_item['attention_mask'].view(
                                    batch_size*(num_negative), -1).to(device)
                }

        q_inputs = {'input_ids': question_item['input_ids'].to(device),
                'attention_mask': question_item['attention_mask'].to(device)
                }
                                            # (batch_size, 10, 768) 0: positive 1~9: negative // [:, 0, :] / [:, 1:, :]
        # .view(batch_size*(num_negative), -1)

        # positive_inputs = {'input_ids': passage_item['input_ids'][:, 0, :].view(batch_size, -1).to(device), # (batch_size * 4, 384)  =
        #                 'attention_mask': passage_item['attention_mask'][:, 0, :].view(batch_size, -1).to(device)
        #         }
        # gg_idx = np.random.randint(0, 10)
        # negative_inputs = {'input_ids': torch.reshape(passage_item['input_ids'][:, gg_idx, :], (batch_size, -1)).to(device), # (batch_size * 4, 384)  =
        #                 'attention_mask': torch.reshape(passage_item['attention_mask'][:, gg_idx, :], (batch_size, -1)).to(device) #.view(batch_size*(num_negative-1), -1)
        #         }
        
        # for i in range(batch_size):
        #     # print(q_inputs['input_ids'][i])
        #     # print(q_inputs['attention_mask'][i])
        #     print(tokenizer.decode(q_inputs['input_ids'][i]))
        #     print('-'*100)
        #     # print(positive_inputs['input_ids'][i])
        #     # print(positive_inputs['attention_mask'][i])
        #     print(tokenizer.decode(positive_inputs['input_ids'][i]))
        #     print('-'*100)
        #     # print(negative_inputs['input_ids'][i])
        #     # print(negative_inputs['attention_mask'][i])
        #     print(tokenizer.decode(negative_inputs['input_ids'][i]))
        #     print('-'*100)
        #     print('')
        #     print('')
        # exit(0)
        p_outputs = p_encoder(**p_inputs) #(batch_size * 11 * 768)
        q_outputs = q_encoder(**q_inputs)

        # positive_outputs = p_encoder(**positive_inputs) #(batch_size * 11 * 768),
        # negaitive_outputs = p_encoder(**negative_inputs) #(batch_size * 33 * 768) #(num_neg+1), emb_dim) # (batch_size * 4, 768) 
        #q_outputs = q_encoder(**q_inputs)  #(batch_size*, emb_dim)              (batch_size, 1, 768) 
        

        # Calculate similarity score & loss
        # positive_outputs = positive_outputs.view(batch_size, 768) #torch.transpose( p_outputs.pooler_output.view(batch_size, -1, 768), 1 , 2) # (batch_size, 4, 768) =>  (batch_size, 768, 4)
        # negaitive_outputs = negaitive_outputs.view(batch_size, 768)
        #print(p_outputs.size())
        #print(p_outputs[0])
        p_outputs = torch.transpose(p_outputs.pooler_output.view(batch_size, -1, 768), 1 , 2)
        q_outputs = q_outputs.pooler_output.view(batch_size, 1, -1) #batch_size, 1, -1 (batch_size, 768) pooler_output
        
        # sim_scores = torch.matmul(q_outputs, p_outputs)
        # sim_scores = sim_scores.view(batch_size, -1) 
        # print(q_outputs.size())
        # print(p_outputs.size())
        # print(p_outputs[:,:,0])
        # exit(0)
        sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
        #if step == 0: print(sim_scores)
        sim_scores = sim_scores.view(batch_size, -1)
        
        sim_scores = F.log_softmax(sim_scores, dim=1) # 0번 유사도 점수가 높아지도록, 나머지는 낮아지도록 // [1., 0.5, 0.3, 02]
        #if step == 0: print(sim_scores)
        
        loss = F.nll_loss(sim_scores, targets) # 0번이 가장 높게 -> loss 최소화 하도록 학습 [0, 0.5, 0.6, 0.8] 
        #if step == 0: print(loss)
        #loss = criterion(q_outputs, positive_outputs, negaitive_outputs)
        # for pi in range(batch_size):
        #     anchor = q_outputs[pi]
        #     temp_positive = positive_outputs[pi, 0, :]
        #     for ni in range(num_negative-1):
        #         loss += 
        #loss = torch.nn.NLLLoss()(sim_scores, targets)
        
        #break

        loss.backward()        
        optimizer.step()
        optimizer.zero_grad()
        #q_encoder.zero_grad()
        #p_encoder.zero_grad()
        batch_loss += loss.detach().cpu().numpy()
        torch.cuda.empty_cache()
    return batch_loss / len(train_dataloader)

def valid_per_epoch(p_encoder, q_encoder, valid_loader, valid_q_loader, ground_truth):
    print(f'Valid start!')
    with torch.no_grad():
        p_encoder.eval()
        q_encoder.eval()
        
        p_embs = []
        top_1_count = 0
        top_3_count = 0
        top_5_count = 0
        for item in valid_loader:
            p_emb = p_encoder(**item).pooler_output.to('cpu').numpy() # pooler_output
            #p_emb = p_encoder(**item).to('cpu').numpy()
            p_embs.extend(p_emb)
        
        p_embs = torch.Tensor(p_embs).squeeze()
        #print(p_embs.size())
        
        for item, gt in tqdm(zip(valid_q_loader, ground_truth), total = len(valid_q_loader)):
            q_emb = q_encoder(**item).pooler_output.to('cpu')  #(num_query, emb_dim)            # pooler_output
            #q_emb = q_encoder(**item).to('cpu')  #(num_query, emb_dim)            
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))

            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
            
            if gt == rank[0]: top_1_count += 1
            if gt in rank[0:10]: top_3_count += 1
            if gt in rank[0:30]: top_5_count += 1
    
    top_1_acc = top_1_count / len(valid_q_loader)
    top_3_acc = top_3_count / len(valid_q_loader)
    top_5_acc = top_5_count / len(valid_q_loader)

    return top_1_acc, top_3_acc, top_5_acc

if __name__=='__main__':
    main()