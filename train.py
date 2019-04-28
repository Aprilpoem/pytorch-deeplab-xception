import argparse
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from modeling.unet import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from models import segModel

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        net = segModel(self.args,self.nclass)
        net.build_model()
        model = net.model
        optimizer = net.optimizer

        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
            print('weight',weight)
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            #self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)#??
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)


            if isinstance(output,tuple):
                loss = self.criterion(output, target,epoch)
                #print('--------')
                #print('features:',torch.sum(torch.isnan(output[1])))
                #print('scores:', torch.sum(torch.isnan(output[0])))
                output = output[0]
            else:
                loss = self.criterion(output, target)
            if isinstance(loss,tuple):#return many loss value
                loss_sum = loss[0]
                loss1 = loss[1]
                loss2 = loss[2]
                self.writer.add_scalars('train/indi_loss_iter', {'ce':loss1,'triplet':loss2},
                                        i + num_img_tr * epoch)
                self.writer.add_scalar('train/total_loss_iter', loss_sum.item(), i + num_img_tr * epoch)
                loss_sum.backward()
                self.optimizer.step()
                train_loss += loss_sum.item()
            else:
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))


            # Show 10 * 3 inference results each epoch

            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                if self.args.dataset != 'brain':
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
                else:
                    self.summary.visualize_image_four(self.writer, self.args.dataset, image, target, output, global_step,True)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/lr',self.optimizer.param_groups[0]['lr'],epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            #print('val:',image.shape,target.shape)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                if isinstance(output, tuple):
                    #to distinguish triplet loss and other loss
                    loss = self.criterion(output, target, epoch)
                    output = output[0]
                else:
                    loss = self.criterion(output, target)

            #to distinguish ce_dice and other loss
            if isinstance(loss,tuple):
                loss = loss[0]
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))


            #show
            num_img_tr = len(self.val_loader)
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                if self.args.dataset != 'brain':
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
                else:
                    self.summary.visualize_image_four(self.writer, self.args.dataset, image, target, output, global_step)

            #eval
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        is_best = new_pred > self.best_pred
        if epoch>=9 and (epoch+1)%10 == 0:
            if is_best:
                self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

