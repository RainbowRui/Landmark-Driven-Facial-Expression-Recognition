import argparse
import os

class ToolOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        self.initialized = True
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, e.g. 0 0,1,2 0,2, use -1 for CPU')
        parser.add_argument('--mode', type=str, default='train', help='train, validation, test')
        parser.add_argument('--model_path', type=str, default=None, help='the path of pretrained model')
        parser.add_argument('--landmark_num', type=int, default=68, help='the number of landmarks')
        parser.add_argument('--image_width', type=int, default=48, help='the width of input image')
        parser.add_argument('--batch_size', type=int, default=128, help='the batch size of training')
        parser.add_argument('--num_workers', type=int, default=4, help='the number of workers')
        parser.add_argument('--start_lr', type=float, default=1e-2, help='the learning rate of start training')
        parser.add_argument('--learning_rate_decay_start', type=int, default=100, help='the epoch number since learning rate decay start')
        parser.add_argument('--validation_path', type=str, default='./results/validation_', help='the path to save validation result')
        parser.add_argument('--test_path', type=str, default='./results/test_', help='the path to save test result')
        parser.add_argument('--save_path', type=str, default='./model_save/resnet18_', help='the path to save trained model')
        parser.add_argument('--total_epoch', type=int, default=3000, help='the number of total epoch')
        parser.add_argument('--validation_frequency', type=int, default=100, help='the frequency of validation')
        parser.add_argument('--save_frequency', type=int, default=100, help='the frequency of saving model')

        parser.add_argument('--train_image_path', type=str, default='./train_images.txt', help='the path of images for training')
        parser.add_argument('--train_label_path', type=str, default='./train_labels.txt', help='the path of labels for training')
        parser.add_argument('--validation_image_path', type=str, default='./validation_images.txt', help='the path of images for validation')
        parser.add_argument('--validation_label_path', type=str, default='./validation_labels.txt', help='the path of labels for validation')
        parser.add_argument('--test_image_path', type=str, default='./test_images.txt', help='the path of images for testing')

        # related setting
        parser.add_argument('--beta', type=float, default=0.7, help='ratio of high importance group in one mini-batch')
        parser.add_argument('--margin_1', type=float, default=0.5, help='rank regularization margin')
        parser.add_argument('--margin_2', type=float, default=0.4, help='relabeling margin')
        parser.add_argument('--relabel_epoch', type=int, default=1800, help='relabeling samples on each mini-batch after @relabel_epoch epoches')

        return parser

    def gather_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        self.parser = parser

        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt