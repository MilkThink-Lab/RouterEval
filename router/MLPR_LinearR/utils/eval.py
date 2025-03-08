from __future__ import print_function, absolute_import
import torch

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    batch_size = target.size(0)

    argmax = torch.argmax(output, dim=1)

    correct = target[torch.arange(batch_size), argmax]
    
    # print(correct)
    # _, pred = output.topk(maxk, 1, True, True)
    
    # pred = pred.t()
    # correct = target

    # res = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0)
    #     res.append(correct_k.mul_(100.0 / batch_size))
    return correct.mean()