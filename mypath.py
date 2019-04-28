class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '../data/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '../data/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'drive':
            return '../data/DRIVE'
        elif dataset == 'brain':
            return '../data/Brain'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
