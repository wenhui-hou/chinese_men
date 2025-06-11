import pandas as pd
import pickle

from options import args
from utils import get_description_lettics
import os
import time
from transformers import AdamW, get_linear_schedule_with_warmup

dictionary = []
with open('./data/CHIP-CDN/CHIP-CDN_train.json', 'r') as f:
    train_list = json.load(f)
print(len(train_list))#6000
print(train_list[:2]) #list

for i in range(len(train_list)):
    train_list[i]['text'] = _process_single_sentence(train_list[i]['text'], mode="text")
    train_list[i]['normalized_result'] = _process_single_sentence(train_list[i]['normalized_result'], mode='normalized_result')

train_df=pd.DataFrame(train_list)
train_data = train_df.values
train_datas = {}
for i in range(len(train_data)):
    target = []
    golds = train_data[i][1].split("##")
    for gold in golds:
        if gold not in dictionary:
            dictionary.append(gold) 
        target.append(dictionary.index(gold))
        train_datas[train_data[i][0]] = target
print(len(train_datas))
print(len(dictionary))


with open('./data/CHIP-CDN/CHIP-CDN_dev.json', 'r') as f:
    dev_list = json.load(f)
print(len(dev_list))#2000
print(dev_list[:2]) #list
for i in range(len(dev_list)):
    dev_list[i]['text'] = _process_single_sentence(dev_list[i]['text'], mode="text")
    dev_list[i]['normalized_result'] = _process_single_sentence(dev_list[i]['normalized_result'], mode='normalized_result')
dev_df=pd.DataFrame(dev_list)
print(dev_df.head(5))
dev_data = dev_df.values
dev_datas = {}
for i in range(len(dev_data)):
    target = []
    golds = dev_data[i][1].split("##")
    for gold in golds:
        if gold not in dictionary:
            dictionary.append(gold) 
        target.append(dictionary.index(gold))
        dev_datas[dev_data[i][0]] = target
print(len(dev_datas))
print(len(dictionary))

pickle.dump(train_datas, open('./data/CHIP-CDN/train_mentions_gold_CHIP-CDN', 'wb'), -1)
pickle.dump(dev_datas, open('./data/CHIP-CDN/test_mentions_gold_CHIP-CDN', 'wb'), -1)
pickle.dump(dictionary, open('./data/CHIP-CDN/dictionary_CHIP-CDN', 'wb'), -1)


descriptions = list()
with open("./data/CHIP-CDN/dictionary_CHIP-CDN.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        entity = line.strip()
        descriptions.append(entity)
                
print(len(descriptions))

all_tokens, all_masks,all_words_input_cat = get_description_lettics()
print(len(all_tokens)) #9867
print(len(all_masks)) #9867
print(len(all_words_input_cat))#9867  

with open("data/CHIP-CDN/all_dictionary_code_tokens.txt", "w") as file:
    for item in all_tokens:
        file.write(f"{item}\n")
with open("data/CHIP-CDN/all_dictionary_code_masks.txt", "w") as file:
    for item in all_masks:
        file.write(f"{item}\n")
with open("data/CHIP-CDN/all_dictionary_code_lettics-embedding.txt", "w") as file:
    for item in all_words_input_cat:
        file.write(f"{item}\n")