import pickle
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def readfile(path,filelist):
    length = len(filelist)
    #length=30
    files = [os.path.join(path,filelist.iloc[i][0]) for i in range(length)]
    data = {'img':[],'label':[]}
    for ff in files:
        if os.path.exists(ff):
            try:
                brain_data = pd.read_pickle(ff)
            except EOFError:
                print(ff)
                continue
        b_img = brain_data[:,:,:4]
        b_label = brain_data[:,:,-1]
        #process img
        b_img = b_img.astype(np.float)
        b_label = encode_seg(b_label)
        data['img'].append(b_img)
        data['label'].append(b_label)
    return data

def get_dataset(root,filepath,trans=None):
    filelist = pd.read_csv(filepath,header=None)
    #data = readfile(root,filelist)
    files = [os.path.join(root, filelist.iloc[i][0]) for i in range(len(filelist))]
    data_set = Brain(files,trans)
    return data_set

def get_Brain_data():
    path = '../data/Brain/Bra-pickle'
    train_path ='../data/Brain/train.csv'
    valid_path = '../data/Brain/val.csv'

    # train_transform = tfs.Compose([
    #         tfs.RandomVerticalFlip(),
    #         tfs.RandomHorizontalFlip(),
    #         tfs.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1),
    #                      scale=(0.9, 1.1), shear=(-0.2, 0.2)),
    #         ])
    train_transform=None
    train_set = get_dataset(path,train_path,train_transform)
    valid_set = get_dataset(path,valid_path)
    print('train:{},val:{}'.format(len(train_set),len(valid_set)))
    return train_set,valid_set

class Brain(Dataset):
    def __init__(self,filelist,transform=None):
        self.length = len(filelist)
        self.file = filelist
        # self.img_transform = tfs.Compose([
        #     #tfs.ColorJitter(brightness=0.2),
        #     tfs.Normalize([1.0315862, 1.6966758, 1.7888203, 1.707134],
        #                   [0.6971523, 1.5744449, 1.7303163, 1.3844236]),
        #     ])
        #[1.0330547, 1.7065939, 1.7926745, 1.713721] [0.7011645, 1.6004407, 1.7304279, 1.3874912]
        #self.mean = [1.0315862, 1.6966758, 1.7888203, 1.707134],
        #self.std = [0.6971523, 1.5744449, 1.7303163, 1.3844236]
        #compute on train set
        self.means = [0.409633042690903, 0.6010816750664089, 0.35653813817515756, 0.375046598562173]
        self.stds = [0.1797391700474948, 0.17567723960617032, 0.13868687701017868, 0.1620602485055006]
        self.trans = transform
        #self.totensor = tfs.ToTensor()

    def __getitem__(self, item):
        fpath = self.file[item]
        #read file
        brain_data = pd.read_pickle(fpath)
        b_img = brain_data[:, :, :4]
        b_label = brain_data[:, :, -1]
        # process img
        b_img = b_img.astype(np.float)
        b_label = encode_seg(b_label)
        #normalize:
        b_img-=self.means
        b_img/=self.stds
        #transform
        if not self.trans is None:
            b_img = self.trans(b_img)
        image,label = tfs.ToTensor()(b_img),torch.from_numpy(b_label)
        image,label = image.float(),label.float()
        sample = {'image': image, 'label': label}
        return sample

    def __len__(self):
        return self.length


def encode_seg(label):
    seg_label = {0: 0, 1: 1, 2: 2, 4: 3}
    seg_num = [0,1,2,4]
    # for seg in seg_num:
    #     label[label==seg] = seg_label[seg]
    pos = np.where(label==4)
    if len(pos)!=0:
        #print('here!',len(pos))
        label[pos] = 3
    #label = label.astype(np.float)
    return label

def normalize(img,file):
    img = img.astype(float)
    pos = np.where(img == 0)
    if len(pos[0]) == img.size:
        print(file)
    else:
        img[img == 0] = np.nan

        mean = np.nanmean(img,axis=(0,1))
        std = np.nanstd(img,axis=(0,1))
        img = (img - mean) / std
        img[np.isnan(img)] = 0
        print('mean,std',mean,std)

    return img


def compute_ms():
    path='../../../data/Brain/Bra-pickle'
    filepath='../../../data/Brain/train.csv'
    filelist = pd.read_csv(filepath,header=None)
    files = [os.path.join(path, filelist.iloc[i][0]) for i in range(len(filelist))]
    images=[]
    for i,f in enumerate(files):
        print(f)
        data = pd.read_pickle(f)
        img = data[:,:,:4]
        #img = img.astype(np.float32)/255.
        images.append(img)
        # if i>3:
        #     break
    means=[]
    std = []
    images = np.array(images)
    for i in range(4):
        pixels = images[:,:,:,i].ravel()
        pixels = pixels[pixels!=0]
        means.append(np.mean(pixels))
        std.append(np.std(pixels))
    print(means,std)


def spilit_train_val():
    from random import shuffle
    filepath = '/home/wls/svmnt/data/Brain/file.csv'
    zeros = '/home/wls/svmnt/data/Brain/zeros_file.csv'
    filelist = pd.read_csv(filepath, header=None)
    print(len(filelist))
    #zerolist = pd.read_csv(zeros, header=None)
    #zerolist = zerolist.sample(frac=0.001)
    #print(len(zerolist))
    #filelist = pd.merge(filelist, zerolist,how='outer')
    #print(len(filelist))
    #print(tmp)
    # filelist = np.asarray(filelist)
    # zeros_list = np.asarray(tmp)
    # zeros_list = shuffle(zeros_list)
    # print(len(zeros_list))
    # #choose 11 all zeros file
    # filelist = np.vstack((filelist,zeros_list[:11]))
    # filelist = shuffle(filelist)
    filelist = filelist.sample(frac=1)
    print(len(filelist))
    length = len(filelist)
    index = int(length*0.75)
    index_val = int(length * 0.9)
    print(index,index_val-index,length-index_val)
    train = filelist.iloc[:index]
    val = filelist.iloc[index:index_val]
    test = filelist.iloc[index_val:]
    f_train = pd.DataFrame(train)
    f_val = pd.DataFrame(val)
    f_test = pd.DataFrame(test)
    f_train.to_csv('/home/wls/svmnt/data/Brain/train.csv',index=0,header=0)
    f_val.to_csv('/home/wls/svmnt/data/Brain/val.csv',index=0,header=0)
    f_test.to_csv('/home/wls/svmnt/data/Brain/test.csv', index=0, header=0)
    print('done')

def process():
    filepath = '../../../data/Brain/file.csv'
    path = '../../../data/Brain/BraTS17-pickle'
    save = '../../../data/Brain/Bra-pickle'
    filelist = pd.read_csv(filepath, header=None)
    files = [os.path.join(path, filelist.iloc[i][0]) for i in range(len(filelist))]
    #data = {'img': [], 'label': []}
    for i in range(len(filelist)):
        ff = files[i]
        name = filelist.iloc[i][0]
        if os.path.exists(ff):
            try:
                brain_data = pd.read_pickle(ff)
            except EOFError:
                print(ff)
                continue
        b_img = brain_data[:, :, :4]
        b_label = np.expand_dims(brain_data[:, :, -1],-1)
        #print('---',b_img.shape,b_label.shape)
        # process img
        b_img = b_img.astype(np.float)
        minvalue = np.min(b_img,axis=(0,1))
        maxvalue = np.max(b_img,axis=(0,1))
        b_img = (b_img-minvalue)/maxvalue
        #combine
        data = np.dstack((b_img,b_label))
        print(data.shape)
        #data = pd.DataFrame(data)
        #data.to_pickle(os.path.join(save,name))
        pp = os.path.join(save,name)
        print(pp)
        pickle.dump(data,open(pp,'wb'),-1)




def main():
    #process()
    #spilit_train_val()
    compute_ms()
    #preprocess
    #1.normalize
    # path = '/home/wls/svmnt/data/Brain/BraTS17-pickle/'
    # files = glob.glob(path+'*.pickle')
    # print(len(files))
    # solid_file = []
    # all_zeros = []
    # unvalid_file=[]
    # for file in files:
    #     try:
    #         data = pd.read_pickle(file)
    #         #remove all zero
    #         img = data[:,:,4]
    #         img = img.astype(float)
    #         pos = np.where(img == 0)
    #         if len(pos[0]) == img.size:
    #             all_zeros.append(file.split['/'][-1])
    #             continue
    #
    #         solid_file.append(file.split['/'][-1])
    #     except:
    #         print('eoferror',file)
    #         unvalid_file.append(file.split['/'][-1])
    #         continue
    #
    #
    # print('all_zeros{} | unaccess:{}, access:{}'.format(len(all_zeros),len(unvalid_file),len(solid_file)))
    # valid = pd.DataFrame(solid_file)
    # zerosf = pd.DataFrame(all_zeros)
    # valid.to_csv('file.csv',index=0,header=0)
    # zerosf.to_csv('zeros_file.csv',index=0,header=0)


if __name__ == '__main__':
    main()