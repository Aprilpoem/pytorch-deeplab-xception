import torch
import torch.nn as nn
from .iou_loss import _mIoULoss,diceLoss,to_one_hot,_FocalLoss
from .lovasz_softmaxloss import lovasz_hinge,lovasz_softmax

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'bce':
            return self.BCELoss
        elif mode == 'mIou':
            return self.mIouLoss
        elif mode == 'dice':
            return self.DiceLoss
        elif mode == 'lovasz_b':
            return self.lovasz_binary
        elif mode == 'lovasz_m':
            return self.lovasz_multi
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        # if c == 2:
        #     loss = criterion(logit.reshape(n,-1),target.reshape(n,-1).long())
        # if self.batch_average:
        #     loss /= n

        return loss
    
    def BCELoss(self,logit,target):
        criterion = nn.BCELoss(weight=self.weight,reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()
        n, c, h, w = logit.shape
        pred = torch.sigmoid(logit)
        if target.shape[1] != pred.shape[1]:
            target = to_one_hot(target,c)

        assert target.shape[1] == pred.shape[1]

        #print(pred.shape,target.shape)
        preds = pred.reshape(n, -1)
        target = target.reshape(n, -1)
        loss = criterion(preds,target.float())
        
        # if self.batch_average:
        #     loss /= n
        
        return loss
        
    # def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
    #     n, c, h, w = logit.size()
    #     criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
    #                                     size_average=self.size_average)
    #     if self.cuda:
    #         criterion = criterion.cuda()
    #
    #     logpt = -criterion(logit, target.long())
    #     pt = torch.exp(logpt)
    #     if alpha is not None:
    #         logpt *= alpha
    #     loss = -((1 - pt) ** gamma) * logpt
    #
    #     if self.batch_average:
    #         loss /= n
    #
    #     return loss

    def FocalLoss(self,logit,target):
        class_num = logit.shape[1]
        criterion = _FocalLoss(class_num)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit,target)
        return loss

    def mIouLoss(self, logit, target ):
        criterion = _mIoULoss()
        if self.cuda:
            criterion = criterion.cuda()
        if len(target.size()) == 3:
            target = to_one_hot(target,logit.shape[1])
        assert target.shape == logit.shape

        loss = criterion(logit,target.double())
        return loss

    def DiceLoss(self, logit, target ):
        criterion = diceLoss()
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target)
        return loss

    def lovasz_binary(self,logit, target):
        pred = torch.sigmoid(logit)
        c = pred.shape[1]
        if target.shape[1] != pred.shape[1]:
            target = to_one_hot(target, c)
        loss = lovasz_hinge(logit,target,per_image=False)
        return loss

    def lovasz_multi(self,logit,target):
        pred = torch.sigmoid(logit)
        loss = lovasz_softmax(pred,target,ignore=self.ignore_index)
        return loss
if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




