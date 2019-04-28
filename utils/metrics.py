import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from PIL import Image


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.TP = np.array([0]*255)
        self.TN = np.array([0]*255)
        self.FP = np.array([0]*255)
        self.FN = np.array([0]*255)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def DiceCoff(self):
        dice = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        dice = 2*(dice.sum())
        return dice

    def _generate_matrix(self, gt_image, pre_image, mask=None):
        if mask is None:
            mask = (gt_image >= 0) & (gt_image < self.num_class)
        else:
            mask = np.where(mask!=0)
            #print('mask:',mask.size())
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def compute_el(self):
        confusion_matrix = self.confusion_matrix
        diag_el = np.diag(confusion_matrix)#tp
        fp = np.sum(confusion_matrix, axis=0)-diag_el
        fn = np.sum(confusion_matrix,axis=1)-diag_el

        precision = diag_el/(diag_el+fp)
        recall = diag_el/(diag_el+fn)

        P = np.nanmean(precision)
        R = np.mean(recall)

        return P,R,(precision,recall)


    def add_batch(self, gt_image, pre_image,mask=None):
        if isinstance(gt_image,str):
            if not mask is None:
                mask = process(mask)
            pre_im = process(pre_image,thres=128)
            gt_image = process(gt_image)
            self.confusion_matrix += self._generate_matrix(gt_image, pre_im, mask)
            self.compute_Roc(gt_image,pre_image,mask)
        else:
            assert gt_image.shape == pre_image.shape
            self.confusion_matrix += self._generate_matrix(gt_image, pre_image, mask)


    def compute_Roc(self,gt,pre,mask=None):
        #pre is path
        if mask is None:
            mask = (gt >= 0) & (gt < self.num_class)
        else:
            mask = np.where(mask==1)
        img = Image.open(pre)
        img = np.array(img)
        for threshold in range(0,255):
            gt_img = gt[mask]
            prob_img = np.expand_dims((img>=threshold)*1, 0)
            prob_img = prob_img[mask]
            assert gt_img.shape == prob_img.shape
            self.TP[threshold] += (np.sum(prob_img * gt_img))
            self.FP[threshold] += np.sum(prob_img * ((1 - gt_img)))
            self.FN[threshold] += np.sum(((1 - prob_img)) * ((gt_img)))
            self.TN[threshold] += np.sum(((1 - prob_img)) * (1 - gt_img))

    def show_Roc(self):
        tp = self.TP
        tn = self.TN
        fn = self.FN
        fp = self.FP

        tpr = np.sort(tp/(tp+fn))
        fpr = np.sort(fp/(fp+tn))
        #auc
        roc_auc = auc(fpr,tpr)
        #precision
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        the = 128
        precision,recall = precision[the],recall[the]

        #show
        fig, ax = plt.subplots(1, 1)
        ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        #ax.set_title('Receiver operating characteristic example')
        ax.legend(loc="lower right")
        fig.savefig('roc.png')
        plt.close(fig)
        return precision,recall,roc_auc

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def process(data,thres=0):
    img = Image.open(data)
    img = np.array(img)
    img = (img > thres) * 1
    img = np.expand_dims(img, 0)
    #img = torch.from_numpy(img)
    return img





