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
from model_m import embed_net
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
parser.add_argument('--resume', '-r', default='llcm_agw_p4_n8_lr_0.1_seed_0_best.t', type=str,help='resume from checkpoint')
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
parser.add_argument('--trial', default=1, type=int, metavar='t', help='select the number of extracted trials')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = './Datasets/SYSU-MM01/'
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


def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_img_o = np.zeros((ngall, 3, 384, 192))
    gall_img_m = np.zeros((ngall, 3, 384, 192))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            img = net(input, input, input, input, test_mode[0])
            img_o, img_m = torch.chunk(img, 2, 0)
            gall_img_o[ptr:ptr + batch_num, :, :, :] = img_o.detach().cpu().numpy()
            gall_img_m[ptr:ptr + batch_num, :, :, :] = img_m.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_img_o, gall_img_m

def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    qurery_img_o = np.zeros((nquery, 3, 384, 192))
    qurery_img_m = np.zeros((nquery, 3, 384, 192))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            img = net(input, input, input, input, test_mode[1])
            img_o, img_m = torch.chunk(img, 2, 0)
            qurery_img_o[ptr:ptr + batch_num, :, :, :] = img_o.detach().cpu().numpy()
            qurery_img_m[ptr:ptr + batch_num, :, :, :] = img_m.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return qurery_img_o, qurery_img_m



def save_images(images, labels, cams, modality, base_dir='output'):
    """
    将图像保存到指定的目录结构中。
    """
    os.makedirs(base_dir, exist_ok=True)

    for i, (image_array, label, cam) in enumerate(zip(images, labels, cams)):
        img_save_path = os.path.join(base_dir, modality, f'cam{cam}', f'{label}.png')

        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)

        image_array = np.transpose(image_array, (1, 2, 0))  # CHW to HWC
        image = (image_array * 255).astype(np.uint8)

        # 使用cv2保存图像
        cv2.imwrite(img_save_path, image)

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
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=args.trial)

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

    query_img_IR, query_img_ItM = extract_query_feat(query_loader)
    gall_img_VIS, gall_img_VtM = extract_gall_feat(trial_gall_loader)

    print('start extracting')
    #save_images(gall_img_VIS, gall_label, gall_cam, modality='VIS')
    save_images(gall_img_VtM, gall_label, gall_cam, modality='VtM')
    #save_images(query_img_IR, query_label, query_cam, modality='IR')
    save_images(query_img_ItM, query_label, query_cam, modality='ItM')

    print('extracting complete')