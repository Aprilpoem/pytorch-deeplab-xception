import torch
import torch.nn as nn
from .many_loss import _mIoULoss,diceLoss,to_one_hot,_FocalLoss,generalized_dice,ModifiedTripletMarginLoss
from .lovasz_softmaxloss import lovasz_hinge,lovasz_softmax
from .ohem_loss import OhemCrossEntropy2d

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
        elif mode == 'ohem':
            return self.ohem_ce
        elif mode == 'ge-dice':
            return self.generalized_dice_loss
        elif mode == 'ce_dice':
            return self.ce_dice
        elif mode == 'tri':
            return self.triplet
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())
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
        return loss

    def FocalLoss(self,logit,target):
        class_num = logit.shape[1]
        #al = torch.Tensor([0.7,0.3])
        al = 0.3
        gg = 5
        criterion = _FocalLoss(class_num,gamma=gg)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit,target)
        return loss

    def mIouLoss(self, logit, target ):
        nclass = logit.shape[1]
        criterion = _mIoULoss(n_classes=nclass)
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

    def generalized_dice_loss(self,logit, target):
        criterion = generalized_dice(ignore=self.ignore_index)
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

    def ohem_ce(self,logit,target):
        criterion = OhemCrossEntropy2d(thresh=0.7,min_kept=120000,factor=1)
        loss = criterion(logit,target)
        return loss

# sum two or more loss func
    def ce_dice(self,logit,target):
        ce_loss = self.CrossEntropyLoss(logit,target)
        dice_loss = self.DiceLoss(logit,target)
        loss = ce_loss+dice_loss
        return (loss,ce_loss.item(),dice_loss.item())


#metric learning
    def triplet(self,out,target,epoch=None):
        target = target.long()
        if isinstance(out,tuple):
            preds = out[0]
            criterion_tri = ModifiedTripletMarginLoss(margin=1,percent=0.5)
            criterion_ce = nn.CrossEntropyLoss()
            if self.cuda:
                criterion_tri = criterion_tri.cuda()
                criterion_ce = criterion_ce.cuda()
            loss_ce = criterion_ce(preds, target)
            if epoch<0:
                return loss_ce
            else:
                loss_tri = criterion_tri(out,target)
            loss = loss_ce+loss_tri
            return (loss,loss_ce.item(),loss_tri.item())
        else:
            print('wrong loss')
    # def tri_backward(self,logit, target,number = 5000 ):
    #
    #     label_ceil   = 2
    #     triplet_loss = 0
    #     criterion    = TripletMarginLoss( margin=1, percent=0.25 )
    #
    #     for i in range( label_ceil ):
    #         a_location, p_location, n_location = get_triplet_point( \
    #                 true_masks, label = i, number = number )
    #
    #         anchor   = get_tri_vector( pred_triplets, a_location )
    #         positive = get_tri_vector( pred_triplets, p_location )
    #         negative = get_tri_vector( pred_triplets, n_location )
    #
    #         triplet_loss += criterion(anchor, positive, negative) * weight
    #
    #
    #     triplet_loss /= i + 1
    #     optimizer.zero_grad()
    #     triplet_loss.backward(retain_graph=retain_graph)
    #     optimizer.step()
    #     return triplet_loss.item()

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




