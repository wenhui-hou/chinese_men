import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pickle
import random
import pandas as pd
import torch.nn as nn
import json


def train(args, model, optimizer, epoch, gpu, data_loader, all_tokens, all_masks,all_words_input_cat):

    print("EPOCH %d" % epoch)

    losses = []

    model.train()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader) #1000
    for i in tqdm(range(num_iter)):
        
        #!!!
        inputs_id, masks,mention_words_input_cat, gold = next(data_iter)
        inputs_id, masks,mention_words_input_cat = torch.LongTensor(inputs_id), torch.LongTensor(masks),torch.LongTensor(mention_words_input_cat)
        inputs_id, masks,mention_words_input_cat = inputs_id.cuda(gpu), masks.cuda(gpu),mention_words_input_cat.cuda(gpu) 
        loss = model(inputs_id, masks, mention_words_input_cat, gold, all_tokens, all_masks, all_words_input_cat, "train", test_descriptions=None)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         scheduler.step()
        losses.append(loss.item())

    return losses

def test(args, model, fold, gpu, data_loader, all_tokens, all_masks,all_words_input_cat):

    y, yhat, ysort = [], [], []

    model.eval()
    with torch.no_grad():
        #!!!
        if gpu<0:            
            test_tokens, test_masks,all_words_input_cat = torch.LongTensor(all_tokens), \
                                  torch.LongTensor(all_masks),\
                                  torch.LongTensor(all_words_input_cat)  
        else:
            test_tokens, test_masks,all_words_input_cat = torch.LongTensor(all_tokens).cuda(args.gpu), \
                                  torch.LongTensor(all_masks).cuda(args.gpu),\
                                  torch.LongTensor(all_words_input_cat).cuda(args.gpu)
        #!!!    
        descriptions = model.get_descriptions(test_tokens, test_masks, all_words_input_cat)  
        
    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):
        with torch.no_grad():
            #!!！            
            inputs_id, masks,words_input_cat, gold = next(data_iter)
            if gpu<0:
                inputs_id, masks,words_input_cat = torch.LongTensor(inputs_id), torch.LongTensor(masks),torch.LongTensor(words_input_cat)
            else:
                inputs_id, masks,words_input_cat= torch.LongTensor(inputs_id), torch.LongTensor(masks),torch.LongTensor(words_input_cat),
                inputs_id, masks,words_input_cat= inputs_id.cuda(gpu), masks.cuda(gpu),words_input_cat.cuda(gpu)
            #!!!
            y_pred, y_sorted = model(inputs_id, masks,words_input_cat, gold, all_tokens, all_masks, all_words_input_cat,"test", descriptions)

            for sor in y_sorted:
                ysort.append(sor.detach().cpu().numpy())
            for pred in y_pred:
                y.append(pred.detach().cpu().numpy())
            for go in gold:
                yhat.append(go)
      

    dictionary = pickle.load(open("./data/CHIP-CDN/dictionary_CHIP-CDN", 'rb')) 
    if fold == "train":
        with open('./data/CHIP-CDN/CHIP-CDN_train.json', 'r') as f:
            train_list = json.load(f)
        for i in range(len(train_list)):
            train_list[i]['text'] = _process_single_sentence(train_list[i]['text'], mode="text")
            train_list[i]['normalized_result'] = _process_single_sentence(train_list[i]['normalized_result'], mode='normalized_result')
        train_df=pd.DataFrame(train_list)
        train_data = train_df.values
        train_datas = {}            
        for i in range(len(train_data)):
            target = []
            for gold in y[i]:          
                if dictionary[gold] not in target:
                    target.append(dictionary[gold])
            train_datas[train_data[i][0]] = target[:args.train_top_k]
            
    else:
        with open('./data/CHIP-CDN/CHIP-CDN_dev.json', 'r') as f:
            val_list = json.load(f)
        for i in range(len(val_list)):
            val_list[i]['text'] = _process_single_sentence(val_list[i]['text'], mode="text")
            val_list[i]['normalized_result'] = _process_single_sentence(val_list[i]['normalized_result'], mode='normalized_result')

        val_df=pd.DataFrame(val_list)            
        val_data = val_df.values

        test_datas = {}
        generate_scores = {}
        for i in range(len(val_data)):
            scores = []
            target = []
            for j in range(len(y[i])): 
                if dictionary[y[i][j]] not in target:
                    target.append(dictionary[y[i][j]]) 
                    scores.append(ysort[i][j]) 
            test_datas[val_data[i][0]] = target[:args.top_k] 
            generate_scores[val_data[i][0]] = scores[:args.top_k]

        datas = test_datas
        print(len(datas))
        test_wrong_20 = 0
        for i in range(len(datas)): 
            for gold_entity in yhat[i]:
                if dictionary[gold_entity] not in list(datas.values())[i]: 
                    test_wrong_20 += 1 
                    break   
        recall_20= 1 - test_wrong_20 / len(datas)          
        print("recall@20:", 1 - test_wrong_20 / len(datas)) 

        test_wrong_10 = 0
        for i in range(len(datas)): 
            for gold_entity in yhat[i]:
                if dictionary[gold_entity] not in list(datas.values())[i][:10]: 
                    test_wrong_10 += 1 
                    break
        recall_10= 1 - test_wrong_10 / len(datas)            
        print("recall@10:", 1 - test_wrong_10 / len(datas)) 

        test_wrong_5 = 0
        for i in range(len(datas)): 
            for gold_entity in yhat[i]:
                if dictionary[gold_entity] not in list(datas.values())[i][:5]: 
                    test_wrong_5 += 1 
                    break
        recall_5= 1 - test_wrong_5 / len(datas)
        print("recall@5:", 1 - test_wrong_5 / len(datas))     
        print() 
    
    if args.test_model:
        pickle.dump(train_datas, open(args.DATA_DIR+'/'+'train_candidates', 'wb'), -1) 
        pickle.dump(test_datas, open(args.DATA_DIR+'/' + 'test_candidates', 'wb'), -1)
        pickle.dump(generate_scores, open(args.DATA_DIR+'/' + 'generate_scores', 'wb'), -1)          
    return recall_20,recall_10,recall_5
            
def str_q2b(text):
    ustring = text
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def _process_single_sentence(sentence, mode='text'):
    sentence = str_q2b(sentence)
    sentence = sentence.strip('"')
    if mode == "text":
        sentence = sentence.replace("\\", ";")
        sentence = sentence.replace(",", ";")
        sentence = sentence.replace("、", ";")
        sentence = sentence.replace("?", ";")
        sentence = sentence.replace(":", ";")
        sentence = sentence.replace(".", ";")
        sentence = sentence.replace("/", ";")
        sentence = sentence.replace("~", "-")
    return sentence            
