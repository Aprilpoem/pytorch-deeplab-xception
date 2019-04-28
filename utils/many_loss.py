import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def to_one_hot(tensor, nClasses,ignore=None):
    if not ignore is None:
        new_tensor = tensor[tensor!=ignore]
        new_tensor = new_tensor.reshape(-1)
        true_1_hot = torch.eye(nClasses)[new_tensor.long()]#n*h*w,nClasses
        true_1_hot = true_1_hot.permute(1,0).float()
        return true_1_hot.cuda()
    # if no ignore index
    tensor = tensor.squeeze(1)
    if not ignore is None:
        ignore_idx = tensor[tensor==ignore]
        tensor[ignore_idx] = 0
    true_1_hot = torch.eye(nClasses)[tensor.long()]
    #print('onehot',true_1_hot.shape)
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    return true_1_hot.cuda()


class _mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(_mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1).double()

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)

        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union

        ## Return average loss over classes and batch
        return 1-loss.mean()

class diceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True,ignore=None):
        super(diceLoss, self).__init__()
        self.ignore = ignore

    def forward(self,logits, true, eps=1e-7):
        num_classes = logits.shape[1]
        probas = F.softmax(logits, dim=1)
        if len(true.size()) == 3:
            true_one_hot = to_one_hot(true,num_classes,ignore=self.ignore)
        assert probas.shape[1] == true_one_hot.shape[1]

        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_one_hot, dims)
        cardinality = torch.sum(probas + true_one_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)

class generalized_dice(nn.Module):
    def __init__(self,ignore=None,isweighted=True):
        super(generalized_dice, self).__init__()
        self.ignore = ignore
        self.isweight = isweighted

    def forward(self,logits, true, eps=1e-7):
        num_classes = logits.shape[1]
        probas = F.softmax(logits, dim=1)
        probas = probas.transpose(1,0).reshape(num_classes,-1)
        true_flat = true.reshape(-1)
        probas = probas[:,true_flat!=255]
        if len(true.size()) == 3:
            true_one_hot = to_one_hot(true,num_classes,ignore=self.ignore)

        if self.isweight:
            weight = torch.zeros((num_classes,1))
            for i in range(num_classes):
                true = true_one_hot[i,:]
                weight[i] = torch.sum(true)
            weight = weight/torch.sum(weight)
            weight = 1/(weight**2+eps)
            #print('w',weight)
        else:
            weight = torch.ones((num_classes, 1))
        weight = weight.cuda()
        dims = 1
        intersection = weight * torch.sum( probas * true_one_hot, dims)
        cardinality = weight * torch.sum(probas + true_one_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)

class _FocalLoss(nn.Module):
    def __init__(self,class_num,iscuda=True,alpha=None,gamma=1):
        super(_FocalLoss,self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num,1)
        else:
            self.alpha = torch.Tensor([[1-alpha],[alpha]])
        if iscuda:
            self.alpha = self.alpha.cuda()
        self.gamma = gamma

    def forward(self, logits, true, eps = 1e-12):
        logits = torch.clamp(logits,eps)
        probs = F.softmax(logits,dim=1)
        classes = logits.shape[1]
        true_one_hot = to_one_hot(true,classes)
        pt = (probs * true_one_hot).sum(1).view(-1,1)
        logpt = pt.log()

        alpha = self.alpha[true.view(-1).long()]

        loss = -alpha * ((1-pt)**self.gamma)* logpt

        return loss.mean()

def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(probas, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)


#metric ways
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum()  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum()  # .pow(.5)
        losses = F.sigmoid(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class ModifiedTripletMarginLoss(nn.Module):
    """Triplet loss function.
    """
    def __init__(self, margin, norm=2, percent=1 ):
        super(ModifiedTripletMarginLoss, self).__init__()
        self.margin  = margin
        self.percent = percent
        self.norm = norm
        self.pdist   = PairwiseDistance(norm)  # norm 2

    def forward(self,outs,logits):
        preds = outs[0]
        nclass = preds.shape[1]
        features = outs[1]
        #sigmoid
        preds = torch.sigmoid(preds)
        features = torch.sigmoid(features)
        n, c, h, w = features.shape
        self.features = features.permute(0, 2, 3, 1).reshape(n * h * w, c)
        loss = 0
        # for i in range(1,nclass):
        #     self.get_mask(preds,logits,i)
        #     loss += self.generate_triplet(features)
        #loss /= (nclass-1)

        #=====
        #simple triplet
        for i in range(1,nclass):
            flag = self.get_mask(preds,logits,i)
            if flag:
                anchor, positive, negative = self.generate_triplet_simple()
                d_p = self.pdist.forward(anchor, positive)
                dist = torch.clamp(self.margin+d_p - self.pdist.forward(anchor, negative),min=0.0)#dp-dn
                #sample_sort = dist_hinge.sort(descending=True)[0]
                #hard_samples = sample_sort[0:int(len(dist_hinge) * self.percent)]
                loss += torch.mean(dist)
        #loss /= (nclass-1)


        #cross entropy loss
        # labmda = 0.2
        # loss_ce = F.cross_entropy(preds,logits)
        # loss = loss + loss_ce

        return loss

    def get_mask(self,preds,logits,positive_label=1):
        #preds NxCxHxW
        positive_class = positive_label
        # probs,prediction = preds.max(1)
        # meanvalue = torch.mean(probs,dim=(1,2))
        # print(logits.shape)
        # print(meanvalue.shape)
        # print(probs.shape)
        #
        # print(meanvalue)
        # #positive mask & negative mask
        p_mask = torch.where(logits==positive_class,torch.full_like(logits,1),torch.full_like(logits,0))
        n_mask = torch.where(logits != positive_class, torch.full_like(logits,1),torch.full_like(logits,0))
        p_mask_flat = p_mask.reshape(-1)
        n_mask_flat = n_mask.reshape(-1)

        p_num = torch.sum(p_mask_flat)
        n_num = torch.sum(n_mask_flat)
        if p_num == 0 or n_num == 0:
            return False
        print('p{},n{}'.format(p_num,n_num))
        minvalue = min(p_num,n_num)
        print(minvalue)
        #print(p_num,n_num)
        if minvalue <= 1000:
            if p_num > n_num:
                #small p_mask
                p_mask_flat = self.small_mask(p_mask_flat,p_num,n_num)
            elif p_num < n_num:
                frac = n_num//p_num
                #print('frac:',frac)
                #frac = 3 if frac>3 else frac
                frac = 1 if frac > 1 else frac
                n_mask_flat = self.small_mask(n_mask_flat,n_num,p_num*frac)
            else:
                pass
        else:
            num=1000
            p_mask_flat = self.small_mask(p_mask_flat, p_num, num)
            n_mask_flat = self.small_mask(n_mask_flat, n_num, num)
        #low confidence area
        #probs
        #lf_mask = torch.where(probs<meanvalue,torch.full_like(logits,1),torch.full_like(logits,0))
        #self.lp_mask = p_mask.mul(lf_mask)
        #self.ln_mask = n_mask.mul(lf_mask)
        self.lp_mask = p_mask_flat
        self.ln_mask = n_mask_flat
        return True

    def small_mask(self,mask,total_num,num):
        assert total_num > num
        mask = mask.cpu().numpy()
        pos = np.where(mask==1)[0]
        #print(pos.shape)
        #choose num element
        #choose_pos = np.random.choice(pos,size=total_num-num,replace=False)
        start = np.random.randint(0,total_num-num)
        choose_pos=pos[start:start+num]
        mask[choose_pos] = 0
        mask = torch.from_numpy(mask)
        return mask.cuda()

    def generate_triplet(self,features):
        n,c,h,w = features.shape
        self.features = features.permute(0,2,3,1).reshape(n*h*w,c)
        #print('fea',torch.sum(torch.isnan(self.features)))
        # the number of positive and negative in a batch

        #comute distance matrix
        positive_fea = self.features[self.lp_mask==1]
        negative_fea = self.features[self.ln_mask==1]


        #==============================================
        #randomly choose p_sum*n negative element
        # n_random_idx = np.random.choice(n_sum,p_sum*frac,False)
        # n_random_idx = torch.from_numpy(n_random_idx)
        # negative_fea = negative_fea[n_random_idx]
        #==============================================
       
        #p_matrix = euclidean_distances(positive_fea,positive_fea)#numpy nxn (n=p_sum)
        #torch_p_matrix = torch.from_numpy(p_matrix)
        #print('ppp,', torch.sum(torch.isnan(positive_fea)))
        torch_p_matrix = self.pairwise(positive_fea,positive_fea)
        #print('p,',torch.sum(torch.isnan(torch_p_matrix)))
        hard_anchor_positive_dist,_ = torch.max(torch_p_matrix,dim=1)
        del torch_p_matrix

        # n_matrix = euclidean_distances(positive_fea,negative_fea)
        # torch_n_matrix = torch.from_numpy(n_matrix)
        torch_n_matrix = self.pairwise(positive_fea, negative_fea)
        #print('n,', torch.isnan(torch_n_matrix).shape)
        #print(torch_n_matrix.shape)
        hard_anchor_negative_dist,_ = torch.min(torch_n_matrix,dim=1)
        del torch_n_matrix
        
        dist = torch.clamp(self.margin + hard_anchor_positive_dist - hard_anchor_negative_dist, min=0.0)
        #print(torch.isnan(dist).shape)
        loss = torch.mean(dist)
        #print('tri',loss.item())
        return loss

    def generate_triplet_simple(self):

        # comute distance matrix
        positive_fea = self.features[self.lp_mask == 1]
        negative_fea = self.features[self.ln_mask == 1]
        torch_p_matrix = self.pairwise(positive_fea.detach(), positive_fea.detach())
        _, hard_anchor_positive_idx = torch.max(torch_p_matrix, dim=1)
        del torch_p_matrix

        torch_n_matrix = self.pairwise(positive_fea.detach(), negative_fea.detach())
        _, hard_anchor_negative_idx = torch.min(torch_n_matrix, dim=1)
        del torch_n_matrix

        #anchor = torch.arange(positive_fea.shape[0])
        assert hard_anchor_positive_idx.shape == hard_anchor_negative_idx.shape
        return positive_fea,positive_fea[hard_anchor_positive_idx],negative_fea[hard_anchor_negative_idx]

    def pairwise(self,x,y):
        #x^2+y^2-2*x.dot(y)
        eps = 1e-4 / x.shape[1]
        xx = torch.sum(torch.pow(x,2),dim=1)#sample_1
        yy = torch.sum(torch.pow(y,2),dim=1)#sample_2
        xy = x.mm(y.transpose(1,0))#sample_1 x sample2
        dis = torch.unsqueeze(xx,-1)- 2*xy + yy
        dis = torch.clamp(dis, min=0.0)
        #print('is zero',torch.sum(dis<0))
        dis_m = torch.pow(dis + eps, 1. / self.norm)
        return dis_m



class PairwiseDistance(nn.Module):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm)
        out = torch.sum(out,dim=1)
        out = torch.clamp(out, min=0.0)
        return torch.pow(out + eps, 1. / self.norm)
