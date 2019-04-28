import torch
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from mypath import Path
import os
from PIL import Image
from dataloaders.datasets.retina import extract_patches_test,merge_overlap
from dataloaders.datasets.brats import get_dataset
import torchvision.transforms as tfs
from utils.metrics import Evaluator,process
from tqdm import tqdm
import numpy as np
from models import segModel
from torch.utils.data import DataLoader


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
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
class Infer(object):
    def __init__(self,args):
        self.args = args
        self.nclass  = 4
        self.save_fold = 'brain_re/brain_cedice'
        mkdir(self.save_fold)
        self.name = self.save_fold.split('/')[-1].split('_')[-1]
        #===for brain==========================
        # self.nclass = 4
        # self.save_fold = 'brain_re'
        #======================================
        net = segModel(self.args,self.nclass)
        net.build_model()
        model = net.model
        #load params
        resume = args.resume
        self.model = torch.nn.DataParallel(model)
        self.model = self.model.cuda()
        print('==>Load model...')
        if not resume is None:
            checkpoint = torch.load(resume)
            # model.load_state_dict(checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
        self.model = model
        print('==>loding loss func...')
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
        elif self.args.dataset == 'brain':
            path = root_path+'/Bra-pickle'
            valid_path = '../data/Brain/test.csv'
            self.valid_set = get_dataset(path,valid_path)
        print('loading test data...')

        #define data
        self.test_loader = None

    def eval(self):
        gt_name = os.listdir(self.test_label)
        img_list = [os.path.join(self.test_label, image) for image in gt_name]
        mask_listdir = [os.path.join(self.test_mask,image.split('.')[0].split('_')[0]+'_test_mask.gif') for image in gt_name]
        pred_list = get_result_list(gt_name,self.save_fold)
        #transform
        for i in range(len(img_list)):
            target, preds, _mask = img_list[i],pred_list[i],mask_listdir[i]
            self.evaluator.add_batch(target, preds,mask=_mask)
        #idx = len(img_list)
        idx=1
        test_Acc = self.evaluator.Pixel_Accuracy()
        test_acc_class = self.evaluator.Pixel_Accuracy_Class()
        test_mIou = self.evaluator.Mean_Intersection_over_Union()
        test_fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        pre,recall,auc=self.evaluator.show_Roc()
        print('Test:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, precision:{}, Recall:{}, Auc:{}".format(test_Acc / idx, test_acc_class / idx, test_mIou / idx,
                                                                test_fwiou / idx,pre,recall,auc))


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
            full_preds = merge_overlap(preds, Height, Width, stride_h, stride_w)  # Tensor->[1,1,H,W]
            full_preds = full_preds[0,1,:,:]
            full_img = tfs.ToPILImage()((full_preds*255).type(torch.uint8))
            full_image = (full_preds>=0.5)*1#0.5
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
        print(self.model)
        self.evaluator.reset()
        self.test_loader = DataLoader(self.valid_set,batch_size=self.args.test_batch_size,shuffle=False)
        tbar = tqdm(self.test_loader, desc='\r')#need to rewrite
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            #show(image[0].permute(1,2,0).numpy(),target[0].numpy())
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                _,pred = output.max(1)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            target = target.cpu().numpy()
            pred = pred.cpu().numpy()

            #show
            if i >= 0 and i<=100:
                iii = image[0].cpu().numpy()
                showimg = np.transpose(iii,(1,2,0))
                plt.figure()
                plt.imshow(showimg,cmap='gray')
                plt.show()
                fname = self.save_fold+'/'+self.name+'_'+str(i)+'.png'
                show(image[0].permute(1,2,0).cpu().numpy(),target[0],pred[0],fname)
            # if i>99:
            #     break
            #save(pred[0],fname=str(i)+'.jpg')
            #

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        Dice_coff = self.evaluator.DiceCoff()
        P, R, perclass = self.evaluator.compute_el()
        print('Test:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, dice:{}".format(Acc, Acc_class, mIoU, FWIoU,Dice_coff))
        print('precision:{},recall:{}'.format(P,R))
        print('preclass, pre{},recall:{}'.format(perclass[0],perclass[1]))
        print('Loss: %.3f' % test_loss)

def show(img,target,pred,fname):
    plt.figure(dpi=600)
    plt.subplot(131)
    plt.imshow(img[:,:,0],cmap='gray')
    plt.subplot(132)
    plt.imshow(target)
    plt.subplot(133)
    plt.imshow(pred)
    plt.show()
    plt.savefig(fname)
# def save(pred,fname):
#     plt.imsave(fname,pred)

def merge_three(img,target,pred):
    h,w = target.shape[0],target.shape[1]
    #img = (img[:,:,0]*255).astype(np.uint8)
    img = img[:,:,:3]
    #zzz = np.
    #image = np.dstack((img,img,img))
    target = simple_decode(target)
    pred = simple_decode(pred)

    new = np.zeros((h,w*3,3))
    new[:,:w,:] = img
    new[:,w:2*w,:] = target
    new[:,2*w:,:] = pred
    return new




def simple_decode(label_mask):
    n_classes=4
    label_colours = np.array([[0,0,0],[0,0,255],[228,199,16],[0,255,0]])#black,blue,yellow,green
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r/255.
    rgb[:, :, 1] = g/255.
    rgb[:, :, 2] = b/255.
    return rgb

if __name__ == '__main__':
    x = np.zeros((100,100))
    x[50:,:50]=1
    x[:50,50:]=2
    x[50:,50:]=4
    plt.figure()
    plt.imshow(x)
    plt.show()




#visualize cluster of classes

