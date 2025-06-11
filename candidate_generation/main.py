from options import args
import random
import numpy as np
import torch
import csv
import sys
from utils import prepare_instance_bert, MyDataset, my_collate_bert, get_description,get_description_lettics
from models import pick_model
from torch.utils.data import DataLoader
import os
import time
from train_test import train, test
from transformers import AdamW, get_linear_schedule_with_warmup

if __name__ == "__main__":


    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)

    csv.field_size_limit(sys.maxsize)

    model = pick_model(args)

    if args.load_model:
        pretrained_model_path = args.load_model + '/' + model.name
        state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    prepare_instance_func = prepare_instance_bert

  
    train_instances = prepare_instance_func( './data/CHIP-CDN/train_mentions_gold_CHIP-CDN',
                                            args)                          
    print("train_instances {}".format(len(train_instances)))
    test_instances = prepare_instance_func('./data/CHIP-CDN/test_mentions_gold_CHIP-CDN',
                                           args)
    print("test_instances {}".format(len(test_instances)))

    collate_func = my_collate_bert
    train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=collate_func)
    if args.test_model:
        train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=False, collate_fn=collate_func)
    test_loader = DataLoader(MyDataset(test_instances), args.batch_size, shuffle=False, collate_fn=collate_func)

    all_tokens, all_masks = get_description()

    with open(args.DATA_DIR+ '/'+ "all_dictionary_code_tokens.txt", "r") as file:
        lines = file.readlines()
    all_tokens = [list(map(int, line.strip()[1:-1].split(', '))) for line in lines]
    with open(args.DATA_DIR+ '/'+ "all_dictionary_code_masks.txt", "r") as file:
        lines = file.readlines()
    all_masks =  [list(map(int, line.strip()[1:-1].split(', '))) for line in lines]

    with open(args.DATA_DIR+ '/'+ "all_dictionary_code_lattices-embedding.txt", "r") as file:
        lines = file.readlines()
    all_words_input_cat = [list(map(float, line.strip()[1:-1].split(', '))) for line in lines]
    
    dir_name = args.model + args.dir_name
    model_dir = os.path.join(args.MODEL_DIR, dir_name)

    optimizer = AdamW(
        model.parameters(),
        # optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    total_steps = len(train_loader) * args.n_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    test_only = args.test_model is not None

    if not test_only:
        for epoch in range(args.n_epochs):            
            if epoch == 0 and not args.test_model:
                dir_name = args.model + args.dir_name
                model_dir = os.path.join(args.MODEL_DIR, dir_name)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
            
            epoch_start = time.time()
            losses = train(args, model, optimizer, epoch, args.gpu, train_loader, scheduler, all_tokens, all_masks,all_words_input_cat)
            loss = np.mean(losses)
            epoch_finish = time.time()
            print("epoch finish in %.2fs, loss: %.4f" % (epoch_finish - epoch_start, loss))
            
            fold = 'dev'
            # test on dev 
            evaluation_start = time.time()
            test(args, model, fold, args.gpu, test_loader, all_tokens, all_masks,all_words_input_cat)
            evaluation_finish = time.time()
            print("evaluation finish in %.2fs" % (evaluation_finish - evaluation_start))

            current_epoch = epoch  
            checkpoint = {
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, model_dir + '/' + 'medbert_checkpoint_epoch_{}.pt'.format(current_epoch))
            if epoch == args.n_epochs - 1:
                print("last epoch: testing on dev and test sets")
                test(args, model, fold, args.gpu, test_loader, all_tokens, all_masks,all_words_input_cat)   
                torch.save(model.state_dict(), model_dir + '/' + 'medbert')
            
    else:
        model_dir = args.test_model
        pretrained_model_path = args.test_model + '/' + model.name
        state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        test(args, model, "train", args.gpu, train_loader, all_tokens, all_masks,all_words_input_cat)
        test(args, model, "test", args.gpu, test_loader, all_tokens, all_masks,all_words_input_cat)

    sys.stdout.flush()
