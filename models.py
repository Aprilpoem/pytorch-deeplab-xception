from modeling.deeplab import *
from modeling.unet import U_Net,U_Net_
import torch.optim as optim


class segModel(object):
    def __init__(self,args,nclass):
        self.args = args
        self.name = args.model_name
        self.outclass = nclass
        self.model = None
        self.optimizer = None
    def build_model(self,name='deeplab'):
        if self.name == 'deeplab':
            self.model = DeepLab(num_classes=self.outclass,
                            inc=self.args.inchannels,
                            backbone=self.args.backbone,
                            output_stride=self.args.out_stride,
                            sync_bn=self.args.sync_bn,
                            freeze_bn=self.args.freeze_bn,
                                 )
            # model = U_Net()
            # train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.args.lr},
            #                 {'params': self.model.get_10x_lr_params(), 'lr': self.args.lr * 10}]
            train_params = [{'params': self.model.parameters(),'lr': self.args.lr}]

            self.optimizer = optim.SGD(train_params, momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
            #self.optimizer = optim.Adam(train_params,weight_decay=self.args.weight_decay)

        elif self.name == 'unet':
            self.model = U_Net(n_channels=self.args.inchannels,n_classes=self.outclass)
            self.optimizer = optim.SGD(self.model.parameters(),lr=self.args.lr,momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
        elif self.name == 'unet_tri':
            self.model = U_Net_(n_channels=self.args.inchannels, n_classes=self.outclass)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
        else:
            raise NotImplementedError


