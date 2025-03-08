import argsparser
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from utils import Logger, AverageMeter, mkdir_p, accuracy
from torch.utils import data
import models
import tools
import os
from dataloader import DatasetTrain, DatasetTest
import numpy as np

def main(args):

    datadict = np.load(args.datadir)
    #for item in datadict:
    #    print(item, datadict[item].shape)
    X_train = datadict['train_embed']
    Y_train = datadict['train_score']
    X_test = datadict['test_embed']
    Y_test = datadict['test_score']
    
    # define a model
    if args.model == 'linear':
        model = models.Linear(X_train.shape[1], Y_train.shape[1]).cuda()
    elif args.model == 'FNN':
        model = models.FNN(X_train.shape[1], args.hidden_size, Y_train.shape[1]).cuda()
    else:
        model = None
        
    
    #for name, param in model.named_parameters():
    #    print(name, param.size())

    """
    Define Residual Methods and Optimizer
    """

    # Resume
    title = ''
    if args.resume:
        # Load checkpoint.
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['state_dict'])
        
        
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Epoch', 'LR', 'train loss', 'train acc', 'test loss', 'test acc'])
    
    parameters_to_train = [param for name, param in model.named_parameters()]

    lr_scheduler = {'coslr': tools.cosine_lr,
                    'steplr': tools.step_lr,
                    'constant': tools.constant}
    
    criterion = nn.BCEWithLogitsLoss()
    
    params = [{"params": parameters_to_train, "weight_decay": args.weight_decay}]
        
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = None
    
    training_set = DatasetTrain(args.datadir)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=args.train_batch, num_workers=args.workers)
    
    testing_set = DatasetTest(args.datadir)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=10000, num_workers=args.workers)
    
    current_iters = 0

    test_losses = AverageMeter()
    test_acc = AverageMeter()

    for citer, batch in enumerate(testloader):
        loss, acc = tools.test(batch, model, criterion)
        
        test_losses.update(loss, batch[0].size(0))
        test_acc.update(acc, batch[0].size(0))
            

    for epoch in range(args.epoch):
        """
        training
        """
        train_losses = AverageMeter()
        test_losses = AverageMeter()

        train_acc = AverageMeter()
        test_acc = AverageMeter()
        
        for citer, batch in enumerate(trainloader):
            lr = lr_scheduler[args.lr_strategy](optimizer, args.lr, current_iters, len(trainloader) * args.epoch)
            loss, acc = tools.train(batch, model, criterion, optimizer)
           
            train_losses.update(loss, batch[0].size(0))
            train_acc.update(acc, batch[0].size(0))
            
            current_iters += 1
            
        for citer, batch in enumerate(testloader):
            loss, acc = tools.test(batch, model, criterion)
            
            test_losses.update(loss, batch[0].size(0))
            test_acc.update(acc, batch[0].size(0))
            
        
        logger.append([epoch, lr, train_losses.avg, train_acc.avg, test_losses.avg, test_acc.avg])
    predicted_probs = []
    for citer, batch in enumerate(testloader):
        output = tools.predict(batch, model)
        predicted_probs.extend(output.detach().cpu().numpy())
    predicted_probs = np.array(predicted_probs)
    predicted_llm_indices = np.argmax(predicted_probs, axis=1)
    overall_accuracy = np.mean(Y_test[np.arange(Y_test.shape[0]), predicted_llm_indices])
    
    
    print('acc on the test set : {}'.format(overall_accuracy))
    print('router acc / bsm acc: {}'.format(overall_accuracy/np.max(np.mean(Y_test, axis=0))))
    # predicted_probs = predicted_probs / predicted_probs.sum(axis=1, keepdims=True)
    # for i in range(predicted_probs.shape[0]):
    #     if np.isnan(predicted_probs[i, :]).any():
    #         predicted_probs[i, :] = np.ones(predicted_probs.shape[1]) / predicted_probs.shape[1]
    # terms = np.where(predicted_probs > 1e-10, predicted_probs * np.log2(predicted_probs), 0)
    predicted_probs = torch.softmax(torch.from_numpy(predicted_probs), dim=1).numpy()
    terms = predicted_probs * np.log2(predicted_probs)
    Ep = -np.sum(terms) / predicted_probs.shape[0] 

    print('Classification bias : {}'.format(Ep))
    tools.save_checkpoint({'state_dict': model.state_dict()}, checkpoint=args.checkpoint)
    logger.close()

if __name__ == '__main__':
    parser = argsparser.get_argparser()
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #print(args)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    with open(args.checkpoint + "/configs.txt", 'w+') as f:
        for (k, v) in args._get_kwargs():
            f.write(k + ' : ' + str(v) + '\n')

    main(args)
