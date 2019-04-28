from modeling.unet import U_Net_
from tensorboardX import SummaryWriter
from dataloaders.datasets import retina
import torch.nn as nn
from datetime import datetime
import torch
import os
from PIL import Image
import numpy as np
import tqdm

def projector(net,image,full_target, Height=584, Width=565, patch_w=48,patch_h=48,stride_h=12, stride_w=12):
    global writer
    image= image.cuda()
    net.eval()
    with torch.no_grad():
        features = predict_a_batch(model=net,patchs=image,labels=full_target,
                                   ph=patch_h,pw=patch_w)
        #merge features
        #full_preds = retina.merge_overlap(features, Height, Width, stride_h, stride_w)
        #print(full_preds.shape)
        #full = full_preds.permute(0,2,3,1)
        #print(full.shape)
        #full = full_preds.reshape(full.shape[0]*full.shape[1]*full.shape[2],full.shape[3])
        #label = list(full_target.reshape(-1))
        #print(len(label))
        #print('img,label',full.shape,label.shape)
        #writer.add_embedding(full,metadata=label)
    #writer.close()



def predict_a_batch(model,patchs,labels,nclass=2,ph=48,pw=48,batch_size=128):
    global writer
    number_of_patch = patchs.shape[0]
    n,c,h,w = patchs.shape
    f_dims = 64
    results = torch.zeros(number_of_patch,f_dims,ph,pw)#features
    print('result',results.shape)
    results = results.cuda()
    patchs = patchs.float()
    steps = int(number_of_patch/batch_size)


    for i in tqdm.trange(steps):
        if i>0:
            break
        start_index = i * batch_size
        end_index = start_index + batch_size
        input_data = patchs[start_index:end_index]
        label_data = labels[start_index:end_index]
        #print('input',input_data.shape)
        output = model(input_data)
        #print('ff',output.shape)
        features = output[1]
        features = torch.sigmoid(features)
        results[start_index:end_index] = features

        #reshape
        features = features.permute(0,2,3,1)
        print('f',features.shape)
        features = features.reshape(features.shape[0]*h*w,features.shape[3])
        label_ = label_data.reshape(-1)
        print(label_.shape)
        features,label_=generate_new(features,label_)
        writer.add_embedding(features, metadata=label_,global_step=i)

    results[end_index:] = torch.sigmoid(model(patchs[end_index:]))
    return results

def generate_new(features,gt_label):
    gt = gt_label.cpu().numpy()
    features = features.cpu().numpy()
    obj_pos = np.where(gt==1)
    bg_pos = np.where(gt!=1)
    length = obj_pos[0].size
    print('len:',length)
    new_bg_pos = bg_pos[0][:length*2]
    features = np.vstack((features[obj_pos],features[new_bg_pos]))
    gt = np.concatenate((gt[obj_pos],gt[new_bg_pos]))
    return features,gt

if __name__ == '__main__':
    f = str(datetime.now())
    writer = SummaryWriter('cluster/' + f)
    net = U_Net_(n_channels=1,n_classes=2)
    model_path='run/drive/unet-tri/experiment_2/checkpoint_120.pth.tar'
    checkpoint = torch.load(model_path)
    net = net.cuda()
    net.load_state_dict(checkpoint['state_dict'])

    #get data
    test_img = '../data/DRIVE/test/images'
    test_label='../data/DRIVE/test/1st_manual'
    imgs = os.listdir(test_img)
    labels = []
    for i in imgs:
        label_name = (i.split('.')[0]).split('_')[0] + '_manual1.gif'
        labels.append(label_name)
    img_list = [os.path.join(test_img, image) for image in imgs]
    label_list = [os.path.join(test_label, lab) for lab in labels]

    # some params
    patch_h = 48
    patch_w = 48
    stride_h = 5
    stride_w = 5
    # crop imgs to patches
    images_patch, labels_patch, Height, Width, _ = retina.extract_patches_test(img_list, label_list, patch_h,
                                                                                         patch_w, stride_h,
                                                                                         stride_w)  # list[patches]
    #full_label
    idx = 0
    flabel = np.array(Image.open(label_list[idx]))
    flabel = (flabel.astype(float))/255.
    image = images_patch[idx]
    label = labels_patch[idx]

    print(image.shape,label.shape)
    projector(net,image,label,Height, Width,patch_w,patch_h,stride_h,stride_w)
    writer.close()

