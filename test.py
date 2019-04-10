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
from modeling.unet import U_Net

def save(img,fname):
    img.save(fname)


class Infer(object):
    def __init__(self,args):
        self.args = args
        self.nclass  = 2
        #define network
        # model = DeepLab(num_classes=self.nclass,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn)
        model = U_Net(3,2)
        #load params
        resume = args.resume
        self.model = torch.nn.DataParallel(model)
        self.model = self.model.cuda()
        print('==>Load model...')
        checkpoint = torch.load(resume)
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
        patch_h = 128
        patch_w = 128
        stride_h = 100
        stride_w = 100
        #crop imgs to patches
        imgs, labels, Height, Width = extract_patches_test(img_list, label_list, patch_h, patch_w, stride_h,
                                                           stride_w)  # list[patches]
        data = []
        for i, j in zip(imgs, labels):
            data.append((i, j))

        #start test one batch has one image
        tbar = tqdm(data)
        test_loss = 0.0
        for idx,sample in enumerate(tbar):
            image,target = sample[0],sample[1]
            #print(image.shape,target.shape)
            image,target = image.cuda(),target.cuda()
            with torch.no_grad():
                outs = self.model(image)
                #print(outs.shape,target.shape)
            loss = self.criterion(outs,target)
            test_loss += loss.item()

            _, preds = outs.max(1, keepdim=True)  # [n,1,h,w]

            #probs_bg = outs[:, 0, :, :]
            #probs_obj = torch.unsqueeze(outs[:, 1, :, :], 1)

            #merge patches into one image
            #full_probs_obj = merge_overlap(probs_obj, Height, Width, stride_h, stride_w)
            full_preds = merge_overlap(preds, Height, Width, stride_h, stride_w)  # Tensor->[1,1,H,W]
            full_preds = tfs.ToPILImage()(full_preds)

            #save result image
            save(full_preds,os.path.join('result',str(idx)+'.jpg'))
            del full_preds

            #eval
            target = target.cpu().numpy()
            preds = preds.cpu().numpy()
            self.evaluator.add_batch(target, preds)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Test:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % (test_loss/len(data)))

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






