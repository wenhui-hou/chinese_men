import csv
import numpy as np
import pickle
import pandas as pd
from options import args
import json
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time

from transformers import BertTokenizer
from transformers import XLMRobertaTokenizer

def prepare_instance_bert(filename, args):
    instances = []
    if args.bert_dir=="JianglabSSUMI/TeaBERT":
        wp_tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_dir)   
    else:
        wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    mention_gold = pickle.load(open(filename, 'rb'))
    
    #!!!
    save_data_name='data/CHIP-CDN/lattices/save_CHIP-CDN.dset'
    with open(save_data_name, 'rb') as fp:
        data = pickle.load(fp) 
    # print(data.char_alphabet) 

    for mention, gold in mention_gold.items():
        mention_tokens = wp_tokenizer.tokenize(mention)
        tokens_max_len = int(args.mentions_max_length - 2)
        if len(mention_tokens) > tokens_max_len:
            mention_tokens = mention_tokens[:tokens_max_len]
        mention_tokens.insert(0, '[CLS]')
        mention_tokens.append('[SEP]')
        mention_tokens_id = wp_tokenizer.convert_tokens_to_ids(mention_tokens)
        
        #!!!!
        mention_word_input_cat=get_Lattice_input(mention_tokens,data) 
        mention_words_input_cat=torch.mean(mention_word_input_cat, dim=0) 
        mention_words_input_cat = mention_words_input_cat.detach().tolist()
        # print(type(mention_words_input_cat)) #list

        mention_masks = [1] * len(mention_tokens_id)
        mention_padding = [0] * (args.mentions_max_length - len(mention_tokens_id))
        mention_tokens_id += mention_padding
        mention_masks += mention_padding

        dict_instance = {
            'mention_tokens_id': mention_tokens_id,
               'mention_masks': mention_masks,
            'mention_words_input_cat':mention_words_input_cat,
                     'gold': gold,
                        
                        }

        instances.append(dict_instance)

    return instances


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word
def get_Lattice_input(tokens,data,number_normalized=True, model_type='lstm',use_char=False,use_count=True):

    words = []
    chars = []
    word_Ids = []       
    char_Ids = []
        
    for word in tokens:          
        if number_normalized:  
            word = normalize_word(word)  
        words.append(word)            
        word_Ids.append(data.word_alphabet.get_index(word))      
        char_list = []
        char_Id = []
        for char in word:
            char_list.append(char)
        for char in char_list:
            char_Id.append(data.char_alphabet.get_index(char))   
        chars.append(char_list)            
        char_Ids.append(char_Id)                         

                
    gaz_Ids = []
    layergazmasks = []
    gazchar_masks = []
    w_length = len(words)

    gazs = [ [[] for i in range(4)] for _ in range(w_length)]  
    gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]
    gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)] 
    max_gazlist = 0
    max_gazcharlen = 0
    for idx in range(w_length):  
        matched_list = data.gaz.enumerateMatchList(words[idx:]) 
        matched_length = [len(a) for a in matched_list]                     
        matched_Id  = [data.gaz_alphabet.get_index(entity) for entity in matched_list] 

        if matched_length:
            max_gazcharlen = max(max(matched_length),max_gazcharlen)

        for w in range(len(matched_Id)):
            gaz_chars = []
            g = matched_list[w] 
            for c in g: 
                gaz_chars.append(data.word_alphabet.get_index(c))  
            if matched_length[w] == 1: 
                gazs[idx][3].append(matched_Id[w]) 
                gazs_count[idx][3].append(1)
                gaz_char_Id[idx][3].append(gaz_chars)
                
            else:   
                gazs[idx][0].append(matched_Id[w])  
                data.gaz_count[1]=1
                gazs_count[idx][0].append(data.gaz_count[matched_Id[w]]) 
                gaz_char_Id[idx][0].append(gaz_chars)  
                wlen = matched_length[w] 
                gazs[idx+wlen-1][2].append(matched_Id[w]) 
                gazs_count[idx+wlen-1][2].append(data.gaz_count[matched_Id[w]])   
                gaz_char_Id[idx+wlen-1][2].append(gaz_chars)    
                for l in range(wlen-2):
                    gazs[idx+l+1][1].append(matched_Id[w])  
                    gazs_count[idx+l+1][1].append(data.gaz_count[matched_Id[w]])
                    gaz_char_Id[idx+l+1][1].append(gaz_chars)

        for label in range(4):
            if not gazs[idx][label]: 
                gazs[idx][label].append(0)
                gazs_count[idx][label].append(1)
                gaz_char_Id[idx][label].append([0])

            max_gazlist = max(len(gazs[idx][label]),max_gazlist) #
                    
        matched_Id  = [data.gaz_alphabet.get_index(entity) for entity in matched_list] 
        if matched_Id:
            gaz_Ids.append([matched_Id, matched_length])
        else:
            gaz_Ids.append([])
                
    for idx in range(w_length): 
        gazmask = []
        gazcharmask = []
        for label in range(4): 
            label_len = len(gazs[idx][label]) 
            count_set = set(gazs_count[idx][label])
            if len(count_set) == 1 and 0 in count_set: 
                gazs_count[idx][label] = [1]*label_len
            
            mask = label_len*[0]  
            mask += (max_gazlist-label_len)*[1]
                        
            gazs[idx][label] += (max_gazlist-label_len)*[0]  
            gazs_count[idx][label] += (max_gazlist-label_len)*[0]  
            char_mask = []
            for g in range(len(gaz_char_Id[idx][label])):
                glen = len(gaz_char_Id[idx][label][g])
                charmask = glen*[0] 
                charmask += (max_gazcharlen-glen) * [1]
                char_mask.append(charmask)
                gaz_char_Id[idx][label][g] += (max_gazcharlen-glen) * [0]
                        
            gaz_char_Id[idx][label] += (max_gazlist-label_len)*[[0 for i in range(max_gazcharlen)]]
            char_mask += (max_gazlist-label_len)*[[1 for i in range(max_gazcharlen)]]

            gazmask.append(mask)
            gazcharmask.append(char_mask)
        layergazmasks.append(gazmask)
        gazchar_masks.append(gazcharmask)
        
    word_Ids=torch.LongTensor(word_Ids).cuda(args.gpu)
    gazs=torch.LongTensor(gazs).cuda(args.gpu)
    gazs_count=torch.LongTensor(gazs_count).cuda(args.gpu) 
    gaz_char_Id=torch.LongTensor(gaz_char_Id).cuda(args.gpu)
    layergazmasks=torch.LongTensor(layergazmasks).cuda(args.gpu)
    gazchar_masks=torch.LongTensor(gazchar_masks).cuda(args.gpu)
 
    seq_len = word_Ids.size()[0]    
    max_gaz_num = gazs.size(-1)    
    gaz_match = []
        
    word_embedding = nn.Embedding(data.word_alphabet.size(), 50)
    word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
    word_embedding=word_embedding.cuda(args.gpu)
    word_embs = word_embedding(word_Ids)
    
    if model_type != 'transformer':  
        dropout = nn.Dropout(p = 0.5)
        word_inputs_d = dropout(word_embs)   
    else:
        word_inputs_d = word_embs 

   
    if use_char: #False
        gazchar_embeds = word_embedding(gaz_char_Id)

        gazchar_mask = gazchar_masks.unsqueeze(-1).repeat(1,1,1,1,50)
        gazchar_embeds = gazchar_embeds.data.masked_fill_(gazchar_mask.data, 0) 

        gaz_charnum = (gazchar_masks == 0).sum(dim=-1, keepdim=True).float()  #(l,4,gl,1)
        gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
        gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum  #(b,l,4,gl,ce)

        if model_type != 'transformer':
            gaz_embeds = drop(gaz_embeds)
        else:
            gaz_embeds = gaz_embeds

    else: 
        gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), 50)
        gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        gaz_embedding= gaz_embedding.cuda(args.gpu)
        gaz_embeds = gaz_embedding(gazs)  
        model_type='lstm'
        if model_type != 'transformer':
            drop = nn.Dropout(p=0.5) 
            gaz_embeds_d = drop(gaz_embeds)   
        else:
            gaz_embeds_d = gaz_embeds

            
        gaz_mask = layergazmasks.unsqueeze(-1).repeat(1,1,1,50) 
        gaz_mask=gaz_mask.bool()
        
        gaz_embeds = gaz_embeds_d.data.masked_fill_(gaz_mask.data, 0) 

    
    if use_count:
        count_sum = torch.sum(gazs_count, dim=2, keepdim=True) 
        count_sum = torch.sum(count_sum, dim=1, keepdim=True)  
        weights = gazs_count.div(count_sum)  
        weights = weights*4
        weights = weights.unsqueeze(-1)
        gaz_embeds = weights*gaz_embeds 
        gaz_embeds = torch.sum(gaz_embeds, dim=2) 

    else:
        gaz_num = (layergazmasks== 0).sum(dim=-1, keepdim=True).float()  
        gaz_embeds = gaz_embeds.sum(-2) / gaz_num  
        
    gaz_embeds_cat =gaz_embeds.view(seq_len,-1)  
    word_input_cat=gaz_embeds_cat
    
    return word_input_cat 

from torch.utils.data import Dataset
class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def pad_sequence(x, max_len, type=int):

    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    return padded_x


def my_collate_bert(x):
    mention_inputs_id = [x_['mention_tokens_id'] for x_ in x]
    mention_masks = [x_['mention_masks'] for x_ in x]
    #!!!
    mention_words_input_cat = [x_['mention_words_input_cat'] for x_ in x]
    
    gold = [x_['gold'] for x_ in x]
   
    return mention_inputs_id, mention_masks, mention_words_input_cat,gold


def get_positive(targets):
    positive = []
    for target in targets:
        positive += [target] * args.hard
    return positive

def get_negative_hard(targets, model, tokens, masks,words_input_cat):
    global descriptions,negative,target
    negative = []
    hard_number = args.hard #20
    with torch.no_grad():
        tokens, masks,words_input_cat = torch.LongTensor(tokens).cuda(args.gpu), \
                        torch.LongTensor(masks).cuda(args.gpu),\
                         torch.LongTensor(words_input_cat).cuda(args.gpu)
        descriptions = model.get_descriptions(tokens, masks,words_input_cat)
              
        single = []
        for target in targets:
            distance = F.pairwise_distance(descriptions[target], descriptions) 
            sorted, indices = torch.sort(distance, descending=False)
            indices = indices.cpu().numpy().tolist()
            for indice in indices:
                if indice not in targets:
                    single.append(indice)
            single = single[:hard_number]
            negative.extend(single)
    return negative

def get_description_lattices():
    descriptions = list()
    with open("./data/CHIP-CDN/dictionary_CHIP-CDN.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            entity = line.strip()
            descriptions.append(entity)
    if args.bert_dir=="JianglabSSUMI/TeaBERT":
        wp_tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_dir)   
    else:
        wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    
    all_tokens=[]
    all_masks = []
    #!!!
    all_words_input_cat=[]  
    progress_bar = tqdm(total=len(descriptions), desc='Processing Descriptions')

    for description in descriptions:        
        tokens = wp_tokenizer.tokenize(description)
        tokens_max_len = args.candidates_max_length - 2
        if len(tokens) > tokens_max_len:
            tokens = tokens[:tokens_max_len]

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
 
        tokens_id = wp_tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(tokens)
        candidate_padding = [0] * (args.candidates_max_length - len(tokens))
        tokens_id += candidate_padding
        masks += candidate_padding

        save_data_name='data/CHIP-CDN/lettics/save_CHIP-CDN.dset'
        with open(save_data_name, 'rb') as fp:
            data = pickle.load(fp) 
        word_input_cat=get_Lattice_input(tokens,data) 
        words_input_cat=torch.mean(word_input_cat, dim=0) 

        words_input_cat = words_input_cat.detach().tolist()
        all_tokens.append(tokens_id)
        all_masks.append(masks)
        #!!!!
        all_words_input_cat.append(words_input_cat)
        progress_bar.update(1)
    progress_bar.close()    
       
    return all_tokens, all_masks,all_words_input_cat

def get_description():
    descriptions = list()
    with open("./data/CHIP-CDN/dictionary_CHIP-CDN.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            entity = line.strip()
            descriptions.append(entity)

    if args.bert_dir=="JianglabSSUMI/TeaBERT":
        wp_tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_dir)   
    else:
        wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    all_tokens=[]
    all_masks = []
    progress_bar = tqdm(total=len(descriptions), desc='Processing Descriptions')

    for description in descriptions:
        tokens = wp_tokenizer.tokenize(description)

        tokens_max_len = args.candidates_max_length - 2
        if len(tokens) > tokens_max_len:
            tokens = tokens[:tokens_max_len]

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
 
        tokens_id = wp_tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(tokens)
        candidate_padding = [0] * (args.candidates_max_length - len(tokens))
        tokens_id += candidate_padding
        masks += candidate_padding

        all_tokens.append(tokens_id)
        all_masks.append(masks)

        progress_bar.update(1)

    progress_bar.close()    
       
    return all_tokens, all_masks


