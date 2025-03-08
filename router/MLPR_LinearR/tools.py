import math
import torch
import os
import numpy as np
from utils import Logger, AverageMeter, mkdir_p, accuracy
import time
import copy

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

def step_lr(opt, base_lr, e, epochs):
    lr = base_lr
    if e>=epochs*0.5 and e<=epochs*0.75:
        lr = base_lr/10
    elif e>=epochs*0.75:
        lr = base_lr/100
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr
 
def constant(opt, base_lr, e, epochs):
    for param_group in opt.param_groups:
        param_group["lr"] = base_lr
    return base_lr

def train(batch, model, criterion, optimizer):
    model.train()

    x, y = batch
    # print(xi[0])
    x =  x.cuda()
    
    x = torch.autograd.Variable(x.float())
    
    true = y.cuda()
    true = true.float()

    outputs = model(x)

    loss = criterion(outputs, true)

    # print(outputs.data.size(), true.data.size())
    acc = accuracy(outputs.data, true.data)
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    LR = optimizer.param_groups[0]['lr']

    return loss.item(), acc.item()

def test(batch, model, criterion):
    
    model.eval()
    
    
    with torch.no_grad():
        x, y = batch
        x =  x.cuda().float()
        
        true = y.cuda().float()
        true = true.float()
        
        outputs = model(x)
        #print(outputs)
        #print(torch.argmax(outputs,dim=1))
        # print(outputs.size(), true.size())
        loss = criterion(outputs, true)
        acc = accuracy(outputs.data, true.data)
       
    return loss.item(), acc.item()

def predict(batch, model):
    model.eval()
    
    
    with torch.no_grad():
        x, y = batch
        x =  x.cuda().float()
        
        true = y.cuda().float()
        true = true.float()
        
        outputs = model(x)
    return outputs

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
