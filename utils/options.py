import argparse
import os
from utils.tools import mkdirs


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, logger):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.logger = logger

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
        parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
        parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

        parser.add_argument("--vis_port", default='8097', help='port used by visdom.')
        parser.add_argument('--dataroot', default='../data/origin', help='path to images (should have subfolders train, test and val)')
        parser.add_argument('--dataset_name', required=True, help='dataset name')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_id', type=str, default='0', help='gpu id: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--load', '-f', type=str, help='Load model from a .pth file')
        parser.add_argument("--optimizer", type=str, default="sgd")
        # model parameters
        parser.add_argument('--model', type=str, default='resnet18', help='chooses which model to use. [resnet10 | resnet18]')
        # parser.add_argument('--num_classes', type=int, default=4, help='# segmentation result classes')

        # parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')

        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        # dataset parameters
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--image_size', type=int, default=448, help='scale images to this size and throw them into neural networks')
        parser.add_argument('--patch_size', type=int, default=64, help='crop images to this size and throw them into neural networks')
        parser.add_argument('--loss', type=str, default='focal', help='loss function. [bce | focal]')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        self.logger.logger.info(f"{message}\n")

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        mkdirs(expr_dir)
        if not self.isTrain:
            res_dir = os.path.join(opt.results_dir, opt.dataset_name, opt.name)
            mkdirs(res_dir)

        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.world_size = opt.gpus*opt.nodes
        self.print_options(opt)
        self.opt = opt
        return self.opt


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser

class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch', type=int, default=200, help='training epochs')
        parser.set_defaults(phase = "train")
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='exp', help='learning rate policy. [exp | poly | step | plateau]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain = True
        return parser
