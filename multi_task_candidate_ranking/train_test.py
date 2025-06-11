import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pickle
import random

def train(args,data, model, optimizer, epoch, gpu, data_loader, scheduler):
    global loss
    print("EPOCH %d" % epoch)

    losses = []
    
    model.train()
    fgm = FGM(model)    
    pgd = PGD(model)
    K = 3
   
    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):

        inputs, inputs_id, segment_ids, masks, labels1, labels2 = next(data_iter)

        inputs_id, segment_ids, masks, labels1, labels2 = torch.LongTensor(inputs_id), torch.LongTensor(segment_ids),\
                                                          torch.LongTensor(masks), torch.tensor(labels1), torch.tensor(labels2)

        inputs_id, segment_ids, masks, labels1, labels2 = inputs_id.cuda(gpu), segment_ids.cuda(gpu), masks.cuda(gpu),\
                                                          labels1.cuda(gpu), labels2.cuda(gpu)

        
        if args.adv=='FGM':
            num_pred, nen_pred, loss = model(data, inputs, inputs_id, segment_ids, masks, labels1, labels2, "train")
            optimizer.zero_grad()
            loss.backward()            
            fgm.attack()
            num_pred, nen_pred, loss_adv = model(data, inputs,inputs_id, segment_ids, masks, labels1, labels2, "train")
            loss_adv.backward() 
            fgm.restore() 
            optimizer.step()
            scheduler.step()
            losses.append(loss_adv.item())
        
        elif args.adv=='PGD':
            num_pred, nen_pred, loss = model(data, inputs, inputs_id, segment_ids, masks, labels1, labels2, "train")
            optimizer.zero_grad()
            loss.backward()                      
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) 
                if t != K-1:
                    optimizer.zero_grad()
                else:
                    pgd.restore_grad()
                num_pred, nen_pred, loss_adv = model(data, inputs, inputs_id, segment_ids, masks, labels1, labels2, "train")
                loss_adv.backward() 
            pgd.restore() 
            optimizer.step()
            scheduler.step()
            losses.append(loss_adv.item())
            
        elif args.adv=='FreeLB':        
            optimizer.zero_grad() 
            freeLB_instance = FreeLB(adv_K=3, adv_lr=1e-2, adv_init_mag=2e-2, adv_max_norm=0, adv_norm_type='l2', base_model='bert')
            num_pred, nen_pred, loss = freeLB_instance.attack(model, data, inputs,inputs_id, segment_ids, masks, labels1, labels2, "train", gradient_accumulation_steps=1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
        else:
            num_pred, nen_pred, loss = model(data, inputs, inputs_id, segment_ids, masks, labels1, labels2, "train")
            optimizer.zero_grad()
            loss.backward()                     
            optimizer.step()  
            scheduler.step()
            losses.append(loss.item())

    return losses

def test(args,data, model, fold, gpu, data_loader):

    y, yhat, num_y, num_yhat = [], [], [], []

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in tqdm(range(num_iter)):
        with torch.no_grad():
            inputs, inputs_id, segment_ids, masks, labels1, labels2, generate_score = next(data_iter)

            inputs_id, segment_ids, masks, labels1, labels2, generate_score = torch.LongTensor(inputs_id), torch.LongTensor(segment_ids),\
                                                              torch.LongTensor(masks), torch.tensor(labels1), torch.LongTensor(labels2), torch.tensor(generate_score)

            inputs_id, segment_ids, masks, labels1, labels2, generate_score = inputs_id.cuda(gpu), segment_ids.cuda(gpu), masks.cuda(gpu),\
                                                              labels1.cuda(gpu), labels2.cuda(gpu), generate_score.cuda(gpu)
            
            inputs_embeds = getattr(model, 'bert').embeddings.word_embeddings(inputs_id)

            num_pred, nen_pred, loss = model(data,inputs, inputs_id, segment_ids, masks, inputs_embeds ,labels1, labels2, fold)

            num_pred = num_pred.detach().cpu().numpy()
            num_pred = np.argmax(num_pred, axis=1)
            num_y.append(num_pred)
            num_yhat.append(labels1.cpu().numpy())

            nen_pred = torch.softmax(nen_pred, dim=-1)
            generate_score = torch.softmax(generate_score, dim=-1)
            generate_score = 1 - generate_score
            nen_pred = nen_pred + generate_score
            nen_pred = nen_pred.detach().cpu().numpy()
            nen_pred = np.argsort(-nen_pred, axis=-1)
            nen_pred_new = []
            for t in range(len(num_pred)):
                nen_pred_new.append(nen_pred[t][:num_pred[t]].tolist())
            y.append(nen_pred_new)
            yhat.append(labels2.cpu().numpy())

    uni_score,mul_score,score = cdn_cls_metric(y, yhat,num_yhat)  
    uni_score_soft,mul_score_soft,score_soft = cdn_cls_metric_soft(y, yhat,num_yhat)
 
    return uni_score,mul_score,score,uni_score_soft,mul_score_soft,score_soft

def cdn_cls_metric(y, yhat,num_yhat):  
    pred, valid, nums = [], [], [] 
    for i in range(len(y)):
        for j in range(len(y[i])):
            valid.append(1)
            if (1 not in yhat[i][j]):
                pred.append(0)
            else:
                y_hat_results = np.where(yhat[i][j] == 1)[0].tolist()
                y_results = y[i][j]
                y_hat_results.sort()
                y_results.sort()
                if y_hat_results == y_results:
                    pred.append(1)
                else:
                    pred.append(0)
            nums.append(num_yhat[i][j])
    uni_pred, uni_valid = [], [] 
    mul_pred, mul_valid = [], []
    for i in range(len(nums)):
        if nums[i] == 1:
            uni_pred.append(pred[i])
            uni_valid.append(valid[i])
        else:
            mul_pred.append(pred[i])
            mul_valid.append(valid[i])

    uni_score = metrics.accuracy_score(uni_valid, uni_pred)
    mul_score = metrics.accuracy_score(mul_valid, mul_pred)
    score = metrics.accuracy_score(valid, pred)

    print("uni_score, mul_score, score")
    print("%.4f, %.4f, %.4f" % (uni_score, mul_score, score))
    print()
    return uni_score,mul_score,score

def cdn_cls_metric_soft(y, yhat,num_yhat):  
    scores, nums = [], [] 
    for i in range(len(y)):
        for j in range(len(y[i])):
            y_hat_result = np.where(yhat[i][j] == 1)[0].tolist()
            count_intersection = len(set(y[i][j]) & set(y_hat_result))
            count_y = len(y[i][j])
            count_y_hat_result = len(y_hat_result)
            s=count_intersection / max(count_y, count_y_hat_result)
            scores.append(s)        
            nums.append(num_yhat[i][j])
    uni_scores = [] 
    mul_scores = []
    for i in range(len(nums)):
        if nums[i] == 1:
            uni_scores.append(scores[i])
        else:
            mul_scores.append(scores[i])

    uni_score =np.mean(uni_scores)
    mul_score =np.mean(mul_scores)
    score = np.mean(scores)
    
    print("uni_score, mul_score, score")
    print("%.4f, %.4f, %.4f" % (uni_score, mul_score, score))
    print()
    return uni_score,mul_score,score


def cdn_num_metric(num_y,num_yhat):         
    number_pred, num_valid = [], []
    for i in range(len(num_y)):
        for j in range(len(num_y[i])):
            number_pred.append(num_y[i][j])
            num_valid.append(num_yhat[i][j])
    num_uni_pred, num_uni_valid = [], [] 
    num_mul_pred, num_mul_valid = [], [] 
    for i in range(len(num_valid)):
        if num_valid[i] == 1:
            num_uni_pred.append(number_pred[i])
            num_uni_valid.append(num_valid[i])
        else:
            num_mul_pred.append(number_pred[i])
            num_mul_valid.append(num_valid[i])

    num_uni_score_a = metrics.accuracy_score(num_uni_valid, num_uni_pred)
    (num_uni_score_p,num_uni_score_r,num_uni_score_f,num_uni_sup) = metrics.precision_recall_fscore_support(y_pred=num_uni_pred, y_true=num_uni_valid, average='weighted') 
    num_mul_score_a = metrics.accuracy_score(num_mul_valid, num_mul_pred)
    (num_mul_score_p,num_mul_score_r,num_mul_score_f,num_mul_sup) = metrics.precision_recall_fscore_support(y_pred=num_mul_pred, y_true=num_mul_valid, average='weighted')
    num_score_a = metrics.accuracy_score(num_valid, number_pred)
    (num_score_p,num_score_r,num_score_f,num_sup) = metrics.precision_recall_fscore_support(y_pred=number_pred, y_true=num_valid, average='weighted')

    return num_uni_score_a,num_uni_score_p,num_uni_score_r,num_uni_score_f, num_mul_score_a,num_mul_score_p,num_mul_score_r,num_mul_score_f, num_score_a,num_score_p,num_score_r,num_score_f



class FGM():
    global name, param
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        global name, param
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) 
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

        
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
                
                
class FreeLB(object):
    def __init__(self, adv_K=3, adv_lr=1e-2, adv_init_mag=2e-2, adv_max_norm=1.0, adv_norm_type='l2', base_model='bert'):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag   
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model
    def attack(self,model, data, inputs,inputs_id, segment_ids, masks, labels1, labels2, fold, gradient_accumulation_steps):
        input_ids = inputs_id
        embeds_init = getattr(model, 'bert').embeddings.word_embeddings(input_ids)
        if self.adv_init_mag > 0:   
            input_mask = masks.to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2": 
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)  
        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs_embeds = delta + embeds_init  
            inputs_id = None
            num_pred, nen_pred, loss = model(data, inputs,inputs_id, segment_ids, masks,inputs_embeds, labels1, labels2, fold) 
            loss = loss.mean()  
            loss = loss / gradient_accumulation_steps
            loss.backward(retain_graph=True) 
            delta_grad = delta.grad.clone().detach()  
            
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1) 
                denorm = torch.clamp(denorm, min=1e-8) 
                delta = (delta + self.adv_lr * delta_grad / denorm).detach() 
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))

            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
        # return loss, logits
        return num_pred, nen_pred, loss
    