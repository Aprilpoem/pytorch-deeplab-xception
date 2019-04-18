import torch
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from mypath import Path
import os
from PIL import Image
from dataloaders.datasets.retina import extract_patches_test,merge_overlap
import torchvision.transforms as tfs
from utils.metrics import Evaluator
from tqdm import tqdm
import numpy as np
from models import segModel

from modeling.unet import U_Net

import matplotlib.pyplot as plt
import cv2

def save(img,fname):
    img.save(fname)
def tonumpy(img):
    if not isinstance(img,torch.Tensor):
        #for label
        img = np.array(img)
        img[img==255] = 1
        img = np.expand_dims(img,0)
    else:
        img = img.numpy()
    return img
def filesave(img,filename):
    im = Image.fromarray(img)
    im.save(filename)
    # import scipy.misc
    # scipy.misc.imsave(filename, img)
    #cv2.imwrite(filename,img)

def get_result_list( gtname, path):
    #21_manual1.gif
    problist = []
    for i in range(len(gtname)):
        image_order = gtname[i].split('.')[0].split('_')[0]
        problist.append( os.path.join(path, image_order+"_test_prob.bmp" ) )
    return problist

def process(data,thres=0):
    img = Image.open(data)
    img = np.array(img)
    img = (img > thres) * 1
    img = np.expand_dims(img, 0)
    #img = torch.from_numpy(img)
    return img

def merge(original,binary_img):#tensor
    assert len(original.shape)==2
    H,W = original.shape[0],original.shape[1]
    gray_original = tfs.ToPILImage()((original*255).type(torch.uint8))#to Image
    binary_img = tfs.ToPILImage()((binary_img*255).type(torch.uint8))
    Imagefile=[gray_original,binary_img]
    mergeImage = Image.new(binary_img.mode,(W*2,H))
    for i,im in enumerate(Imagefile):
        mergeImage.paste(im,box=(i*W,0))
    return mergeImage

class Infer(object):
    def __init__(self,args):
        self.args = args
        self.nclass  = 2
        self.save_fold = 'result/result_lov'
        net = segModel(self.args,self.nclass)
        net.build_model()
        model = net.model
        #load params
        resume = args.resume
        self.model = torch.nn.DataParallel(model)
        self.model = self.model.cuda()
        print('==>Load model...')
        checkpoint = torch.load(resume)
        #model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model
        self.criterion = SegmentationLosses(cuda=args.cuda).build_loss(mode=args.loss_type)

        #define evaluator
        self.evaluator = Evaluator(self.nclass)

        #get data path
        root_path = Path.db_root_dir(self.args.dataset)
        if self.args.dataset == 'drive':
            folder = 'test'
            self.test_img = os.path.join(root_path, folder, 'images')
            self.test_label = os.path.join(root_path, folder, '1st_manual')
            self.test_mask = os.path.join(root_path, folder, 'mask')

        #define data
        self.test_loader = None
    def eval(self):
        gt_name = os.listdir(self.test_label)
        img_list = [os.path.join(self.test_label, image) for image in gt_name]
        pred_list = get_result_list(gt_name,self.save_fold)
        test_loss = 0.0
        test_Acc = 0.
        test_mIou = 0.
        test_acc_class = [0.]*self.nclass
        test_fwiou = 0.
        #transform
        for i in range(len(img_list)):
            target = process(img_list[i])
            preds = process(pred_list[i],thres=128)
            self.evaluator.add_batch(target, preds)
            # # Fast test during the training
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

            test_Acc += Acc
            test_acc_class += Acc_class
            test_mIou += mIoU
            test_fwiou += FWIoU

            self.evaluator.reset()

        idx = len(img_list)
        print('Test:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(test_Acc / idx, test_acc_class / idx, test_mIou / idx,
                                                                test_fwiou / idx))


    def predict_a_patch(self):
        self.model.eval()
        imgs = os.listdir(self.test_img)
        labels = []
        for i in imgs:
            label_name = (i.split('.')[0]).split('_')[0]+'_manual1.gif'
            labels.append(label_name)
        img_list = [os.path.join(self.test_img,image) for image in imgs]
        label_list = [os.path.join(self.test_label,lab) for lab in labels]

        #some params
        patch_h = self.args.ph
        patch_w = self.args.pw
        stride_h = self.args.sh
        stride_w = self.args.sw
        #crop imgs to patches
        images_patch, labels_patch, Height, Width,self.gray_original = extract_patches_test(img_list, label_list, patch_h, patch_w, stride_h,
                                                           stride_w)  # list[patches]
        data = []
        for i, j in zip(images_patch, labels_patch):
            data.append((i, j))

        #start test one batch has one image
        tbar = tqdm(data)
        for idx,sample in enumerate(tbar):
            image,target = sample[0],sample[1]
            #print(image.shape,target.shape)
            image,target = image.cuda(),target.cuda()
            with torch.no_grad():
                result = self._predict_a_patch(image)
            preds = result
            #print('preds:',preds[preds<0].shape)
            full_preds = merge_overlap(preds, Height, Width, stride_h, stride_w)  # Tensor->[1,1,H,W]
            #print('fullpreds:', full_preds[full_preds < 0].shape)
            full_img = tfs.ToPILImage()((full_preds*255).type(torch.uint8))
            full_image = (full_preds>=0.5)*1
            mergeImage = merge(self.gray_original[idx],full_image)
            #save result image
            name_probs = imgs[idx].split('.')[0].split('_')[0]+'_test_prob.bmp'
            name_merge = imgs[idx].split('.')[0].split('_')[0]+'_merge.bmp'
            save(mergeImage,os.path.join(self.save_fold,name_merge))
            save(full_img,os.path.join(self.save_fold,name_probs))

    def _predict_a_patch(self, patchs):
        number_of_patch = patchs.shape[0]
        results = torch.zeros(number_of_patch,self.nclass,
                            self.args.ph, self.args.pw)
        results = results.cuda()
        patchs = patchs.float()

        steps  = int(number_of_patch / self.args.batch_size)
        #step  = tqdm(steps)
        for i in range(steps):
            start_index = i*self.args.batch_size
            end_index   = start_index + self.args.batch_size
            output  = self.model( patchs[start_index:end_index] )
            output      = torch.sigmoid( output )
            results[start_index:end_index] = output
        results[end_index:] = torch.sigmoid(self.model(patchs[end_index:]))
        return results

    def test(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')#need to rewrite
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            #for retina dataset, target is 4 dims
            if target.ndim == 4:
                target = target[:,1,:,:]
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Test:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)






