from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import scipy.io
import os
from sklearn import manifold, datasets
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
import numpy as np
from PIL import Image
import build_transforms

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='sysu_base_p4_n4_lr_0.1_seed_0_epoch_80.t', type=str,help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='../save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--img_path', default='img/', type=str, help='img save path')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=20, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=30, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='select the number of extracted trials')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
parser.add_argument('--modal', default='all', type=str, help='VIS-IR or VIS-ItM or VtM-IR or VtM-ItM')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = './SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
pool_dim = 2048
print('==> Building model..')
net = embed_net(n_class, arch=args.arch)
net.to(device)
cudnn.benchmark = True


checkpoint_path = args.model_path

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_map = 0  # best test map
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = build_transforms.test_transforms(
    args.img_h, args.img_w, normalize)
transform_color1 = build_transforms.train_transforms_color1(
    args.img_h, args.img_w, normalize)
transform_color2 = build_transforms.train_transforms_color2(
    args.img_h, args.img_w, normalize)
transform_thermal1 = build_transforms.train_transforms_thermal1(
    args.img_h, args.img_w, normalize)
transform_thermal2 = build_transforms.train_transforms_thermal2(
    args.img_h, args.img_w, normalize)
transform_train = transform_color1, transform_color2, transform_thermal1, transform_thermal2

if not os.path.isdir('./save_tsne'):
    os.makedirs('./save_tsne')

result = scipy.io.loadmat('tsne.mat')

query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]

gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

gallery_feature1, gallery_feature2 = np.array_split(gallery_feature, 2, axis=0)
gallery_label1, gallery_label2 = np.array_split(gallery_label, 2, axis=0)

query_feature1, query_feature2 = np.array_split(query_feature, 2, 0)
query_label1, query_label2 = np.array_split(query_label, 2, 0)

def GenIdx(train_color_label, train_thermal_label):
    color_pos = {}  # 使用字典来存储标签号和对应的索引列表
    unique_label_color = np.unique(train_color_label)

    # 为颜色标签创建索引字典
    for label in unique_label_color:
        color_pos[label] = [k for k, v in enumerate(train_color_label) if v == label]

    thermal_pos = {}  # 使用字典来存储标签号和对应的索引列表
    unique_label_thermal = np.unique(train_thermal_label)

    # 为热成像标签创建索引字典
    for label in unique_label_thermal:
        thermal_pos[label] = [k for k, v in enumerate(train_thermal_label) if v == label]

    return color_pos, thermal_pos

class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels

        query_feature = torch.FloatTensor(result['query_f'])
        query_cam = result['query_cam'][0]
        query_label = result['query_label'][0]

        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_cam = result['gallery_cam'][0]
        gallery_label = result['gallery_label'][0]

        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        gallery_feature1, gallery_feature2 = np.array_split(gallery_feature, 2, axis=0)
        gallery_label1, gallery_label2 = np.array_split(gallery_label, 2, axis=0)

        query_feature1, query_feature2 = np.array_split(query_feature, 2, 0)
        query_label1, query_label2 = np.array_split(query_label, 2, 0)

        self.train_color_image1 = gallery_feature1
        self.train_color_label = gallery_label1
        self.train_color_image2 = gallery_feature2

        self.train_thermal_image1 = query_feature1
        self.train_thermal_label = query_label1
        self.train_thermal_image2 = query_feature2

        # BGR to RGB
        self.transform_color1, self.transform_color2, self.transform_thermal1, self.transform_thermal2 = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img10, target10 = self.train_color_image1[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img11, target11 = self.train_color_image2[self.cIndex[index]], self.train_color_label[self.cIndex[index]]

        img20, target20 = self.train_thermal_image1[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img21, target21 = self.train_thermal_image2[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]


        return img10, img11, img20, img21, target10, target20

    def __len__(self):
        return len(self.train_color_label)

end = time.time()

if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

tsne_class = len(np.unique(trainset.train_color_label))

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(tsne_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(tsne_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(n_class,arch=args.arch)
net = net.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))


def plot_embedding(X, y, z, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(15, 12), dpi=100)

    r = []
    #print(X.shape[0])
    for i in range(X.shape[0]):
        if z[i] == 1:
            # print(i, y[i])
            plt.scatter(X[i, 0], X[i, 1], s=120, color=color[y[i]], edgecolor='black', marker='o')
        else:
            plt.scatter(X[i, 0], X[i, 1], s=120, color=color[y[i]], edgecolor='black', marker='^')

    plt.xlim((-100, 100))
    plt.ylim((-100, 100))
    plt.tick_params(axis='x', which='major', labelbottom=False, labelsize=12)


def plot_embedding_all(X, y, z, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    x1, x2 = np.array_split(X, 2, axis=0)
    y1, y2 = torch.chunk(y, 2, 0)
    plt.figure(figsize=(15, 12), dpi=100)

    #print(X.shape[0])
    for i in range(x1.shape[0]):
        if z[i] == 1:
            # print(i, y[i])
            plt.scatter(x1[i, 0], x1[i, 1], s=120, color=color[y1[i]], edgecolor='black', marker='o')
            plt.scatter(x2[i, 0], x2[i, 1], s=120, color=color[y2[i]], edgecolor='black', marker='d')
        else:
            plt.scatter(x1[i, 0], x1[i, 1], s=120, color=color[y1[i]], edgecolor='black', marker='s')
            plt.scatter(x2[i, 0], x2[i, 1], s=120, color=color[y2[i]], edgecolor='black', marker='^')


color = [
    (0.9, 0.2, 0.3, 1.0),  # 淡红色
    (0.3, 0.6, 0.9, 1.0),  # 浅蓝色
    (0.8, 0.4, 0.7, 1.0),  # 紫罗兰
    (0.5, 0.5, 0.0, 1.0),  # 橄榄绿
    (0.2, 0.7, 0.4, 1.0),  # 森林绿
    (0.9, 0.6, 0.1, 1.0),  # 金
    (0.4, 0.3, 0.8, 1.0),  # 深蓝色
    (0.6, 0.8, 0.3, 1.0),  # 橄榄黄
    (0.1, 0.8, 0.7, 1.0),  # 海蓝
    (0.7, 0.3, 0.5, 1.0),  # 玫瑰红
    (0.4, 0.5, 0.9, 1.0),  # 浅紫色
    (0.3, 0.2, 0.7, 1.0),  # 深紫色
    (0.6, 0.4, 0.2, 1.0),  # 棕色
    (0.1, 0.5, 0.2, 1.0),  # 橄榄绿
    (0.9, 0.7, 0.5, 1.0),  # 浅棕色
    (0.3, 0.8, 0.6, 1.0),  # 浅绿色
    (0.7, 0.6, 0.9, 1.0),  # 浅灰色
    (0.2, 0.3, 0.5, 1.0),  # 深蓝色
    (0.8, 0.5, 0.3, 1.0),  # 橙色
    (0.5, 0.1, 0.7, 1.0),  # 亮紫色
    (0.3, 0.6, 0.2, 1.0),  # 黄绿色
    (0.9, 0.1, 0.5, 1.0),  # 亮粉色
    (0.4, 0.8, 0.9, 1.0),  # 浅青色
    (1.0, 0.5, 0.0, 1.0),
]


def train(epoch):
    # current_lr = adjust_learning_rate(optimizer, epoch)
    data_time = AverageMeter()
    # switch to train mode
    net.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (input10, input11, input20, input21, label1, label2) in enumerate(trainloader):
            z1 = torch.ones(label1.shape)
            z2 = torch.zeros(label2.shape)
            z = torch.cat((z1, z2), 0)
            print(batch_idx)

            input10 = Variable(input10.cuda())
            input11 = Variable(input11.cuda())
            input20 = Variable(input20.cuda())
            input21 = Variable(input21.cuda())
            if args.mode == 'all':
                labels = torch.cat((label1, label1, label2, label2), 0)
                z = torch.cat((z1, z1, z2, z2), 0)
                input1 = torch.cat((input10, input11), 0)
                input2 = torch.cat((input20, input21), 0)
            elif args.mode == 'VIS-IR':
                labels = torch.cat((label1, label2), 0)
                z = torch.cat((z1, z2), 0)
                input1 = input10
                input2 = input20
            elif args.mode == 'VIS-ItM':
                labels = torch.cat((label1, label2), 0)
                z = torch.cat((z1, z2), 0)
                input1 = input10
                input2 = input21
            elif args.mode == 'VtM-IR':
                labels = torch.cat((label1, label2), 0)
                z = torch.cat((z1, z2), 0)
                input1 = input11
                input2 = input20
            elif args.mode == 'VtM-ItM':
                labels = torch.cat((label1, label2), 0)
                z = torch.cat((z1, z2), 0)
                input1 = input11
                input2 = input21

            a = labels.unique()
            for i in range(len(a)):
                for j in range(len(labels)):
                    if labels[j] == a[i]:
                        labels[j] = i
            # print(labels)
            data_time.update(time.time() - end)

            out = torch.cat((input1, input2), 0)

            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            X_tsne = tsne.fit_transform(out.detach().cpu().numpy())
            if args.mode == 'all':
                plot_embedding_all(X_tsne, labels, z)
            else:
                plot_embedding(X_tsne, labels, z)
            plt.savefig(os.path.join('save_tsne', 'tsne_{}.jpg'.format(batch_idx)))

import numpy as np
from torch.utils.data import Sampler

class IdentitySampler(Sampler):
    def __init__(self, color_pos, thermal_pos, num_pos, batch_size):
        self.color_pos = color_pos
        self.thermal_pos = thermal_pos
        self.num_pos = num_pos
        self.batch_size = batch_size
        self.num_classes = len(color_pos)
        self.indices = self._generate_indices()

    def _generate_indices(self):
        all_indices = []
        # 直接使用color_pos的键作为迭代器
        for class_idx in self.color_pos.keys():
            class_color_indices = np.random.choice(self.color_pos[class_idx], self.num_pos, replace=True)
            class_thermal_indices = np.random.choice(self.thermal_pos[class_idx], self.num_pos, replace=True)
            # 将颜色和热成像索引配对
            for color_idx, thermal_idx in zip(class_color_indices, class_thermal_indices):
                all_indices.append((color_idx, thermal_idx))
        # 打乱所有索引
        return all_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    # training

print('==> Start Training...')
print('==> Preparing Data Loader...')
# identity sampler

epoch = 0

print(trainset.train_color_label)
sampler = IdentitySampler(color_pos, thermal_pos, args.num_pos, args.batch_size)

# 将sampler的index1和index2属性赋给trainset的cIndex和tIndex
trainset.cIndex = [idx[0] for idx in sampler.indices]  # 颜色图像的索引
trainset.tIndex = [idx[1] for idx in sampler.indices]  # 热成像图像的索引
print(epoch)
print(trainset.cIndex)
print(trainset.tIndex)

loader_batch = args.batch_size * args.num_pos

trainloader = data.DataLoader(trainset, batch_size=loader_batch,
                              sampler=None, num_workers=args.workers, drop_last=True)
# training

train(epoch)
