from __future__ import print_function
import argparse
import time
from PIL import Image
import os
import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from model import embed_net
from utils import *
import pdb
import scipy.io
import build_transforms
import cv2

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
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=0, type=int, metavar='t', help='select the number of extracted trials')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
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

end = time.time()

def process_sysu(data_path, mode, relabel=False):
    if mode == 'rgb':
        cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'ir':
        cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files.extend(new_files)
    img = []
    id = []
    cam = []
    for img_path in files:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        img.append(img_path)
        id.append(pid)
        cam.append(camid)
    return img, np.array(id), np.array(cam)

def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat1 = np.zeros((ngall, 2048))
    gall_feat2 = np.zeros((ngall, 2048))
    gall_feat_att1 = np.zeros((ngall, 2048))
    gall_feat_att2 = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, input, input, test_mode[0])
            feat1, feat2 = torch.chunk(feat, 2, 0)
            feat_att1, feat_att2 = torch.chunk(feat_att, 2, 0)
            gall_feat1[ptr:ptr + batch_num, :] = feat1.detach().cpu().numpy()
            gall_feat2[ptr:ptr + batch_num, :] = feat2.detach().cpu().numpy()
            gall_feat_att1[ptr:ptr + batch_num, :] = feat_att1.detach().cpu().numpy()
            gall_feat_att2[ptr:ptr + batch_num, :] = feat_att2.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat1, gall_feat2, gall_feat_att1, gall_feat_att2

def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat1 = np.zeros((nquery, 2048))
    query_feat2 = np.zeros((nquery, 2048))
    query_feat_att1 = np.zeros((nquery, 2048))
    query_feat_att2 = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, input, input, test_mode[1])
            feat1, feat2 = torch.chunk(feat, 2, 0)
            feat_att1, feat_att2 = torch.chunk(feat_att, 2, 0)
            query_feat1[ptr:ptr + batch_num, :] = feat1.detach().cpu().numpy()
            query_feat2[ptr:ptr + batch_num, :] = feat2.detach().cpu().numpy()
            query_feat_att1[ptr:ptr + batch_num, :] = feat_att1.detach().cpu().numpy()
            query_feat_att2[ptr:ptr + batch_num, :] = feat_att2.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat1, query_feat2, query_feat_att1, query_feat_att2

if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        # model_path = checkpoint_path + 'sysu_awg_p4_n8_lr_0.1_seed_0_best.t'
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_sysu(data_path, mode='ir')
    gall_img, gall_label, gall_cam = process_sysu(data_path, mode='rgb')

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    gall_vis, gall_vtm, gall_vis_att, gall_vtm_att = extract_gall_feat(trial_gall_loader)
    query_ir, query_itm, query_ir_att, query_itm_att = extract_query_feat(query_loader)

    gall_feat = np.concatenate((gall_vis_att, gall_vtm_att), axis=0)
    gall_label = np.concatenate((gall_label, gall_label), axis=0)
    gall_cam = np.concatenate((gall_cam, gall_cam), axis=0)

    query_feat = np.concatenate((query_ir_att, query_itm_att), axis=0)
    query_label = np.concatenate((query_label, query_label), axis=0)
    query_cam = np.concatenate((query_cam, query_cam), axis=0)

    result = {'gallery_f': gall_feat, 'gallery_label': gall_label, 'gallery_cam': gall_cam,
              'query_f': query_feat, 'query_label': query_label, 'query_cam': query_cam
              }
    scipy.io.savemat('tsne.mat', result)