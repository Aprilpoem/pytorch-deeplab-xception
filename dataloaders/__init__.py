#from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, retina
from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, retina
from torch.utils.data import DataLoader
import torchvision.transforms as tfs

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'drive':
        num_class = 2
        train_path='../data/DRIVE/training/images'
        train_mask = '../data/DRIVE/training/1st_manual'
        test_path = '../data/DRIVE/test/'#images and 1st_manual
        simple_transform = tfs.Compose([
            tfs.ToTensor()
        ])
        train_ipatches, train_lpatches, valid_ipatches, valid_lpatches = retina.getdata(train_path,train_mask)
        test_patches,test_mask_patches = retina.getdata(test_path+'images' ,test_path+'1st_manual',False)
        train_set = retina.Retinal(train_ipatches, train_lpatches, transform=simple_transform)
        valid_set = retina.Retinal(valid_ipatches, valid_lpatches, transform=simple_transform)
        test_set = retina.Retinal(test_patches,test_mask_patches,simple_transform)
        train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=False)
        return train_loader, valid_loader,test_loader,num_class
    else:
        raise NotImplementedError

