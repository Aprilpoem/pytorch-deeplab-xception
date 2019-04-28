import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tfs
from torchvision.transforms import functional
import torchvision
import math
import torch.nn.functional as F
from utils.precess import PreProc
import matplotlib.pyplot as plt

#import cv2

def getdata(img_path,mask_path,ph,pw,npatches,valid=True):
    images, labels = read(img_path,mask_path)
    if valid:
        # ph = 48
        # pw = 48
        # npatches = 120000
        images_patches, labels_patches = extarct_patches_train(images, labels,ph,pw,npatches)
        split_percent = 0.2
        valid_num = int(npatches * 0.2)
        n = valid_num
        train_ipatches, train_lpatches = images_patches[:-n], labels_patches[:-n]
        valid_ipatches, valid_lpatches = images_patches[-n:], labels_patches[-n:]

        #print
        print('all patch num:{} | valid:{} | train:{}'.format(len(images_patches),
                                                              valid_num, npatches - valid_num))
        return train_ipatches,train_lpatches,valid_ipatches,valid_lpatches
    else:
        ph = 20
        pw = 20
        sh = 10
        sw = 10
        images_patches, labels_patches ,_,_= extract_patches_test(images, labels, ph, pw, sh, sw)
        return images_patches,labels_patches

def getData(img_list,label_list):
  img_data = []
  label_data = []
  for i in range(len(img_list)):
      img = img_list[i]
      label_name = label_list[i]

      #read image and convert into numpy format
      # WxHx3 -> HxWx3
      img_matrix = Image.open(img).convert('RGB')
      img_matrix = np.array(img_matrix)

      #read segmentation images
      # WxH -> 1xHxW
      label_matrix = Image.open(label_name)
      label_matrix = np.array(label_matrix)
      label_matrix = np.expand_dims(label_matrix,axis=0)

      #add to data matrix
      img_data.append(img_matrix)
      label_data.append(label_matrix)
  return img_data,label_data

def read(img_path,label_path):
    img_list = sorted(os.listdir(img_path))
    label_list = sorted(os.listdir(label_path))
    for i in range(len(img_list)):
        if isinstance(img_list[i],str):
            img_list[i] = img_path + '/'+img_list[i]
            label_list[i] = label_path +'/' + label_list[i]
    return img_list, label_list

def extarct_patches_train(img_list,label_list,patch_h,patch_w,n_patches):#PIL Image format
    length = len(img_list)
    k = n_patches // length
    img_data = []
    label_data = []
    for i in range(len(img_list)):
        img = Image.open(img_list[i])
        H,W = img.size[0],img.size[1]
        label = Image.open(label_list[i])
        #preprocess PIL image
        img = np.asarray(img)
        #print(img.shape)
        data = np.expand_dims(np.transpose(img,(2,0,1)),0)
        gray_img = PreProc(data)
        #To Image
        data = np.transpose(np.squeeze(gray_img,0), (1,2,0))
        #print(type(data))
        # plt.figure()
        # plt.imshow(data[:,:,0])
        # plt.show()
        img =Image.fromarray(data[:,:,0])
        #img.show()
        #img = data

        cnt = 0
        while cnt<k:
            #patch_data,patch_label=rand_crop(img,label,patch_h,patch_w)
            i, j, th, tw = tfs.RandomCrop.get_params(img, (patch_h,patch_w))
            patch_data = functional.crop(img, i, j, th, tw)
            patch_label = functional.crop(label, i, j, th, tw)
            if not isRemove(patch_label):
                continue
            img_data.append(patch_data)
            label_data.append(patch_label)
            cnt += 1
    return img_data,label_data

def isRemove(label):
    #label :PIL.Image (if all black them remove this patch)
    label = tfs.ToTensor()(label)
    assert len(label.shape) == 3
    tot_num = label.shape[1]*label.shape[2]
    pos = label[label==0].shape
    if tot_num == pos[0]:
        return False
    else:
        return True

def rand_crop(data,label,height,width):
    i, j, th, tw = tfs.RandomCrop.get_params(data,(height,width))
    data= functional.crop(data,i,j,th,tw)
    label = functional.crop(label,i,j,th,tw)
    return data,label

def extract_patches_test(img_list, label_list, patch_h, patch_w,stride_h,stride_w):
    img_data = []
    label_data = []
    original_img = []
    H,W=0,0
    for i in range(len(img_list)):
        img = Image.open(img_list[i]).convert('RGB')
        label = Image.open(label_list[i])#read gif
        #============preprocess================
        img = np.asarray(img)
        #print(img.shape)
        data = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
        gray_img = PreProc(data)
        #To tensor
        img = torch.from_numpy(gray_img[0,:,:,:])
        #print(img.shape)
        # To Image
        #data = np.transpose(np.squeeze(gray_img, 0), (1, 2, 0))
        #img = Image.fromarray(data[:, :, 0])
        #======================================
        label = tfs.ToTensor()(label)
        if i == 0:
            H,W = img.shape[1],img.shape[2]
        crop_imgs = crop_with_overlap(img,patch_h,patch_w,stride_h,stride_w)
        crop_labels = crop_with_overlap(label,patch_h,patch_w,stride_h,stride_w)
        img_data.append(crop_imgs)
        label_data.append(crop_labels)

        #original
        original_img.append(img[0,:,:])
    return img_data,label_data,H,W,original_img

def crop_squares(data,h,w):
    #PIL.Image->Tensor
    data = tfs.ToTensor()(data.convert('RGB'))
    C,H,W = data.shape
    #print(C,H,W)
    m = int(math.ceil(H/h))
    n = int(math.ceil(W/w))
    re = torch.randn(m*n,C,h,w)
    #padding:
    H_offset = m*h-H
    W_offset = n*w-W
    if H_offset % 2 == 0:
        top_bottom = (H_offset//2,H_offset//2)
    else:
        top_bottom = (H_offset//2,H_offset//2+1)
    if W_offset % 2 == 0:
        left_right = (W_offset//2,W_offset//2)
    else:
        left_right = (W_offset // 2, W_offset // 2 + 1)
    data = F.pad(data,left_right+top_bottom)
    #print('data:',data.shape)
    x_corner = [h*i for i in range(m)]
    y_corner = [w*i for i in range(n)]
    #print(x_corner,y_corner)
    count = 0
    for x in x_corner:
        for y in y_corner:#from left to right,from up to down
            re[count] = data[:,x:x+h,y:y+w]
            count += 1
    return re

def merge_img(labels,H,W):
    n_patchs,c,h,w = labels.shape
    m = int(math.ceil(H / h))
    #print(m)
    n = int(math.ceil(W / w))
    #print(m,n)
    img = torch.empty(c, m*h, n*w)
    x_corner = [h * i for i in range(m)]
    y_corner = [w * i for i in range(n)]
    for i in range(m):
       for j in range(n):
           img[:,i*h:(i+1)*h,j*w:(j+1)*w]=labels[i*n+j]
    #print(img.shape)
    image = img[:,0:H,0:W]
    return image#Tensor


def crop_with_overlap(imgs,patch_h,patch_w,stride_h,stride_w):
    #imgs.shape-->3 dim
    channel,full_H,full_W = imgs.shape
    h_leftpixels = (full_H-patch_h)%stride_h
    w_leftpixels = (full_W-patch_w)%stride_w
    h_nPatches = (full_H-patch_h)//stride_h+1
    w_nPatches = (full_W-patch_w)//stride_w+1
    nums_of_patches = h_nPatches*w_nPatches#one img could extract how many patches
    #print(nums_of_patches)
    patch_imgs = torch.zeros(nums_of_patches,channel,patch_h,patch_w)
    imgs = F.pad(imgs,(0,stride_h-h_leftpixels,0,stride_w-w_leftpixels))#right,bottom add padding 0
    tot = 0
    #for i in range(imgs.shape[0]):
    for h in range(h_nPatches):
        for w in range(w_nPatches):
            patch_image = imgs[:,h*stride_h:(h*stride_h+patch_h),w*stride_w:(w*stride_w+patch_w)]
            patch_imgs[tot] = patch_image
            tot += 1
    return patch_imgs

def merge_overlap(patch_imgs,full_H,full_W,stride_h,stride_w):
    assert (len(patch_imgs.shape)==4)
    if isinstance(patch_imgs,np.ndarray):
        patch_imgs = torch.from_numpy(patch_imgs).cuda()
    #patch_imgs-->4 dimension->[patch_nums,channels,h,w]
    patch_h = patch_imgs.shape[-2]
    patch_w = patch_imgs.shape[-1]
    h_nPatches = (full_H - patch_h) // stride_h + 1
    w_nPatches = (full_W - patch_w) // stride_w + 1
    h_leftpixels = (full_H - patch_h) % stride_h
    w_leftpixels = (full_W - patch_w) % stride_w
    padding_h = stride_h-h_leftpixels
    padding_w = stride_w-w_leftpixels
    patchs = h_nPatches*w_nPatches

    tot = 0
    #for i in range(nums_of_imgs):
    tmp_image = torch.zeros(1,patch_imgs.shape[1],full_H+padding_h,full_W+padding_w)
    tmp_sum = torch.zeros(1,patch_imgs.shape[1],full_H+padding_h,full_W+padding_w)
    for h in range(h_nPatches):
        for w in range(w_nPatches):
            tmp_image[:,:,h*stride_h:(h*stride_h+patch_h),w*stride_w:(w*stride_w+patch_w)] += patch_imgs[tot].cpu().float()
            tmp_sum[:,:,h*stride_h:(h*stride_h+patch_h),w*stride_w:(w*stride_w+patch_w)] += 1
            tot += 1
    print(tot,patchs)
    tmp_image /= tmp_sum
    result_imgs = tmp_image[:,:,:full_H,:full_W]
    #result_imgs = result_imgs.max(1)[1]
    #result_imgs = result_imgs[0,1,:,:]
    return result_imgs


#check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False

class Retinal(Dataset):
    def __init__(self,imgs,masks,transform=None):
        self.image = imgs
        self.label = masks
        self.length = len(imgs)
        self.trans = transform

    def __getitem__(self, idx):
      image = self.image[idx]#PIL
      target = self.label[idx]#PIL
      #image -> tensor
      img = self.trans(image)
      target = self.encode_seg(target)
      #target = self.trans(target)
      #print('target,',target.shape)
      #target = torch.squeeze(target,0)

      sample = {'image': img, 'label': target}
      return sample

    def __len__(self):
        return self.length

    def encode_seg(self,label):#PIL->numpy class=[0,1]
      label = np.array(label)#HxW
      #np_target[np.where(np_target == 255)] = 1.0
      label[label == 255] = 1
      #labels = np.eye(2)[label.astype(np.int8)]
      #true_masks = labels.transpose(2,0,1)
      #np_target = np_target/255.
      #to->tensor
      #np_target = np.expand_dims(np_target,0)
      #tensor_target =  torch.from_numpy(true_masks)
      #tensor_target.long()
      tensor_target = torch.from_numpy(label)
      return tensor_target.long()


