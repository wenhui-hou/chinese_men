import torch.nn as nn
import os
import numpy as np
from transformers import BertModel
import torch.nn.functional as F
import torch
from torch.nn.init import xavier_uniform_ as xavier_uniform
from utils import get_positive
from options import args
import time
import copy
from torch.nn.init import xavier_uniform_ as xavier_uniform
from itertools import chain

class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()

        self.name = args.model
        self.gpu = args.gpu
#         self.loss = nn.CrossEntropyLoss()
        if args.loss == 'CE':
            self.loss_num = nn.CrossEntropyLoss()
            self.loss_nen = nn.CrossEntropyLoss()
        elif args.loss == 'ASL':
            asl_config = [float(c) for c in args.asl_config.split(',')]
            self.loss_num = AsymmetricLoss(gamma_neg=asl_config[0], gamma_pos=asl_config[1],
                                                clip=asl_config[2], reduction=args.asl_reduction)
            self.loss_nen = AsymmetricLoss(gamma_neg=asl_config[0], gamma_pos=asl_config[1],
                                                clip=asl_config[2], reduction=args.asl_reduction)
        elif args.loss == 'ASLO':
            asl_config = [float(c) for c in args.asl_config.split(',')]
            self.loss_num = AsymmetricLossOptimized(gamma_neg=asl_config[0], gamma_pos=asl_config[1],
                                                         clip=asl_config[2], reduction=args.asl_reduction) 
            self.loss_nen = AsymmetricLossOptimized(gamma_neg=asl_config[0], gamma_pos=asl_config[1],
                                                         clip=asl_config[2], reduction=args.asl_reduction) 
        elif args.loss == 'FL':
            focal_config = [float(c) for c in args.focal_config.split(',')]
            self.loss_num = FocalLoss_multiclass( num_class= 8 ,alpha=None, gamma=focal_config[1])
            self.loss_nen = FocalLoss_multiclass( num_class= 2 ,alpha=focal_config[0], gamma=focal_config[1])
#             self.loss_nen = BCEFocalLoss( alpha=focal_config [2],gamma=focal_config[1],reduction=args.focal_reduction)

        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.classifier = nn.Linear(args.embed_size, 2)
        self.num_classifier = nn.Linear(args.embed_size, 8)
        self.num_multihead_attn = nn.MultiheadAttention(args.embed_size, 8)
        self.nen_multihead_attn = nn.MultiheadAttention(args.embed_size, 8)

    def forward(self, data, inputs,input_ids, segment_ids, attention_mask,inputs_embeds, labels1, labels2, fold):

        if fold != "train":
            flat_inputs = list(chain(*inputs))
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
            flat_segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

            x=[]
            for inx in range(len(flat_inputs)):
                x_sublist=get_Lattice_input1(data, flat_inputs[inx],x_char[inx],number_normalized=False,use_count=True)
                x.append(x_sublist)
            x= torch.stack(x)
            mention_num = []
            for i in range(len(flat_segment_ids)):
                n = 0
                for segment_id in flat_segment_ids[i]:
                    if segment_id == 0:
                        n += 1
                    else:
                        break
                mention_num.append(n)
            mention = []
            for i in range(len(mention_num)):
                mention_num[i]
                men = x[i][1: mention_num[i] - 1, 0: x.size(2)]
                men = torch.mean(men, dim=0)
                mention.append(men)
            mention = torch.cat(mention, dim=0)
            mention = mention.view(len(mention_num), -1)
                  
            candidate_num = []
            for i in range(len(flat_segment_ids)):
                n = 0
                for segment_id in flat_segment_ids[i]:
                    if segment_id == 1:
                        n += 1
            #         else:
            #             break
                candidate_num.append(n)
            nen_pooled_out = []
            for i in range(len(mention_num)):
                men = x[i][1: mention_num[i] - 1, 0: x.size(2)]  
                can = x[i][mention_num[i]: mention_num[i]+candidate_num[i]-1, 0: x.size(2)]  
                seq = torch.cat([men , can], dim=0)
                seq = torch.mean(seq, dim=0)
                nen_pooled_out.append(seq)
            nen_pooled_out = torch.cat(nen_pooled_out, dim=0)
            nen_pooled_out = nen_pooled_out.view(len(mention_num), -1)
                            
            num_key = num_value = torch.unsqueeze(mention, dim=0)
            num_query = torch.unsqueeze(nen_pooled_out, dim=0)
            num_outputs, _ = self.num_multihead_attn(num_query, num_key, num_value)
            num_outputs = torch.squeeze(num_outputs)

            nen_key = nen_value = torch.unsqueeze(nen_pooled_out, dim=0)
            nen_query = torch.unsqueeze(mention, dim=0)
            nen_outputs, _ = self.nen_multihead_attn(nen_query, nen_key, nen_value)
            nen_outputs = torch.squeeze(nen_outputs)

            num_logits = self.num_classifier(num_outputs)
            num_logits = num_logits.view(input_ids.size(0), args.test_top_k, -1)
            num_logits = torch.mean(num_logits, dim=1)
            num_loss = self.loss_num(num_logits, labels1)
            
            nen_logits = self.classifier(nen_outputs)
            nen_logits = nen_logits.view(input_ids.size(0) * args.test_top_k, -1)
            labels2 = labels2.view(input_ids.size(0) * args.test_top_k)
            nen_loss = self.loss_nen(nen_logits, labels2)
            nen_logits = nen_logits.view(input_ids.size(0), args.test_top_k, -1)
            labels2 = labels2.view(input_ids.size(0), args.test_top_k)
            nen_logits_pred = torch.softmax(nen_logits, dim=-1)
            nen_logits_pred = torch.index_select(nen_logits_pred, dim=-1, index=torch.LongTensor([1]).cuda(self.gpu))
            nen_logits_pred = nen_logits_pred.view(input_ids.size(0), args.test_top_k)
            loss = args.alpha * num_loss + (1 - args.alpha) * nen_loss
#             loss = num_loss + nen_loss
            return num_logits, nen_logits_pred, loss

        else:
            outs = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask,inputs_embeds=inputs_embeds)
            x_char=outs[0]
            x=[]
            for inx in range(len(inputs)):
                x_sublist=get_Lattice_input1(data, inputs[inx],x_char[inx],number_normalized=False,use_count=True)
                x.append(x_sublist)
            x= torch.stack(x)
            mention_num = []
            for i in range(len(segment_ids)):
                n = 0
                for segment_id in segment_ids[i]:
                    if segment_id == 0:
                        n += 1
                    else:
                        break
                mention_num.append(n)
            mention = []
            for i in range(len(mention_num)):
                mention_num[i]
                men = x[i][1: mention_num[i] - 1, 0: x.size(2)]
                men = torch.mean(men, dim=0)
                mention.append(men)
            mention = torch.cat(mention, dim=0)
            mention = mention.view(len(mention_num), -1)
                              
            candidate_num = []
            for i in range(len(segment_ids)):
                n = 0
                for segment_id in segment_ids[i]:
                    if segment_id == 1:
                        n += 1
            #         else:
            #             break
                candidate_num.append(n)
#             print(candidate_num)
            nen_pooled_out = []
            for i in range(len(mention_num)):
                men = x[i][1: mention_num[i] - 1, 0: x.size(2)]  
                can = x[i][mention_num[i]: mention_num[i]+candidate_num[i]-1, 0: x.size(2)] 
                seq = torch.cat([men , can], dim=0)
                seq = torch.mean(seq, dim=0)
                nen_pooled_out.append(seq)
            nen_pooled_out = torch.cat(nen_pooled_out, dim=0)
            nen_pooled_out = nen_pooled_out.view(len(mention_num), -1)

            num_key = num_value = torch.unsqueeze(mention, dim=0)
            num_query = torch.unsqueeze(nen_pooled_out, dim=0)
            num_outputs, _ = self.num_multihead_attn(num_query, num_key, num_value)
            num_outputs = torch.squeeze(num_outputs)

            nen_key = nen_value = torch.unsqueeze(nen_pooled_out, dim=0)
            nen_query = torch.unsqueeze(mention, dim=0)
            nen_outputs, _ = self.nen_multihead_attn(nen_query, nen_key, nen_value)
            nen_outputs = torch.squeeze(nen_outputs)

            num_logits = self.num_classifier(num_outputs)
            num_loss = self.loss_num(num_logits, labels1)
            nen_logits = self.classifier(nen_outputs)
            nen_loss = self.loss_nen(nen_logits, labels2)
            loss = args.alpha * num_loss + (1 - args.alpha) * nen_loss
#             loss = num_loss + nen_loss
            return num_logits, nen_logits, loss


def pick_model(args):
    if args.model == "bert":
        model = Bert(args)
    else:
        raise RuntimeError("wrong model name")
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def get_Lattice_input1(data,tokens,x,number_normalized=False,use_count=True):
    embedding_dict = {}
    for i, token in enumerate(tokens):
        vector = x[i, :]
        embedding_dict[(i, token)] = vector
    
    words = []
    word_Ids = []       
    for word in tokens:      
        if number_normalized:  
            word = normalize_word(word)  
        words.append(word)
        word_Ids.append(data.word_alphabet.get_index(word))  
    gaz_Ids = []
    layergazmasks = []
    gazchar_masks = []
    w_length = len(words)
    gazs = [ [[] for i in range(4)] for _ in range(w_length)]  
    gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]
    gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)]  
    count_nonmatch = 0
    max_gazlist = 0
    max_gazcharlen = 0
    for idx in range(w_length): 
        matched_list = data.gaz.enumerateMatchList(words[idx:]) 
        matched_length = [len(a) for a in matched_list] 
        matched_Id  = [data.gaz_alphabet.get_index(entity) for entity in matched_list] 
        if matched_length:
            max_gazcharlen = max(max(matched_length),max_gazcharlen)
        for w in range(len(matched_list)):
            gaz_chars = []
            g = matched_list[w] 
            for c in g:
                gaz_chars.append(c)
            if matched_length[w] == 1: 
                gazs[idx][3].append(matched_list[w]) 
                gazs_count[idx][3].append(1)
                gaz_char_Id[idx][3].append(gaz_chars)

            else:   
                gazs[idx][0].append(matched_list[w])  
                data.gaz_count[1]=1
                gazs_count[idx][0].append(data.gaz_count[matched_Id[w]])  
                gaz_char_Id[idx][0].append(gaz_chars) 
                wlen = matched_length[w] 
                gazs[idx+wlen-1][2].append(matched_list[w])  
                gazs_count[idx+wlen-1][2].append(data.gaz_count[matched_Id[w]]) 
                gaz_char_Id[idx+wlen-1][2].append(gaz_chars)  
                for l in range(wlen-2): 
                    gazs[idx+l+1][1].append(matched_list[w]) 
                    gazs_count[idx+l+1][1].append(data.gaz_count[matched_Id[w]]) 
                    gaz_char_Id[idx+l+1][1].append(gaz_chars) 
        for label in range(4):
            if not gazs[idx][label]:
                gazs[idx][label].append(0)
                gazs_count[idx][label].append(1)
                gaz_char_Id[idx][label].append([0])
            max_gazlist = max(len(gazs[idx][label]),max_gazlist)
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
    gazs_count=torch.LongTensor(gazs_count)

    word_Ids=torch.LongTensor(word_Ids)
    seq_len = word_Ids.size()[0]    
    max_gaz_num =len(gazs[0][0])   
    use_count=True
    if use_count: 
        count_sum = torch.sum(gazs_count, dim=2, keepdim=True)  
        count_sum = torch.sum(count_sum, dim=1, keepdim=True)  
        weights = gazs_count.div(count_sum) 
        weights = weights*4
        weights = weights.unsqueeze(-1) 

    gazs_embedding = []
    for inx1 in range(len(gazs)):
        words_in_sentence=gazs[inx1]
        sentence_embeddings = []
        for inx2 in range(len(words_in_sentence)):
            word=words_in_sentence[inx2]
            word_embeddings= []
            for inx3 in range(len(word)):
                w=word[inx3]
                if w == 0 :  
                    w_embedding = torch.ones_like(x[0]) 
                else:
                    w_embeddings=[]
                    inxs=[]
                    for inx4 in range(len(w)):
                        i= w[inx4] 
                        if i == words_in_sentence[3][0]:
                            inx=inx4        
                            inxs.append(inx)
                
                    for inx4 in range(len(w)):
                        i= w[inx4] 
                        if inx2==0:
                            i_embeddings= embedding_dict[(inx1+inx4,i)]                    
                        if inx2==1:
                            if inxs==[]:
                                count_nonmatch += 1
                                i_embeddings= torch.zeros(768).to(args.gpu)
                            elif inxs[0]!=0:
                                i_embeddings= embedding_dict[(inx1-inxs[0]+inx4,i)]  
                            else:
                                i_embeddings= embedding_dict[(inx1-inx+inx4,i)]
                        if inx2==2:
                            i_embeddings= embedding_dict[(inx1-len(w)+1+inx4,i)]                    
                        if inx2==3:
                            i_embeddings= embedding_dict[(inx1,i)]                                        
                        w_embeddings.append(i_embeddings)
                    if w_embeddings:
                        w_embedding = torch.mean(torch.stack(w_embeddings), dim=0)
                    else:
                        w_embedding = torch.ones_like(x[0])
                word_embeddings.append(w_embedding)
            sentence_embeddings.append(word_embeddings)
        gazs_embedding.append(sentence_embeddings)
    gazs_embedding = torch.stack([torch.stack([torch.stack(word_embedding) for word_embedding in sentence_embeddings]) for sentence_embeddings in gazs_embedding])
    if gazs_embedding.device != weights.device:
        weights = weights.to(gazs_embedding.device)
    gaz_embeds = weights*gazs_embedding 
    gaz_embeds = torch.sum(gaz_embeds, dim=2)  
    gaz_embeds_cat = torch.mean(gaz_embeds, dim=1) 
    x_words = torch.mean(torch.stack([x, gaz_embeds_cat]), dim=0)
    return x_words

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,
                 reduction='sum'):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1) 
        # Basic CE calculation 
        y = y.view(-1, 1) 
        idx = y.cpu().long()
        one_hot_key = torch.FloatTensor(y.size(0), x.size(1)).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != xs_pos.device:
            one_hot_key = one_hot_key.to(xs_pos.device)                        
        y=one_hot_key    
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        if self.reduction == 'mean':
            return -loss.mean()
        elif self.reduction == 'sum':
            return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False,
                 reduction='sum'):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        y = y.view(-1, 1) 
        idx = y.cpu().long()
        one_hot_key = torch.FloatTensor(y.size(0), x.size(1)).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != x.device:
            one_hot_key = one_hot_key.to(x.device)                        
        y=one_hot_key
        
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        if self.reduction == 'mean':
            return -self.loss.mean()
        elif self.reduction == 'sum':
            return -self.loss.sum()
    
import torch.nn as nn
import torch.nn.functional as F
    
class FocalLoss_multiclass(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss_multiclass, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
#         print(target[0])#tensor([1])  
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
        
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss    

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self,alpha=0.25, gamma=2, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):       
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        target = target.view(-1, 1) 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), 2).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != pt.device:
            one_hot_key = one_hot_key.to(pt.device)

            
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

        