from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData, LLCMData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
from loss import OriTripletLoss, TcaLoss
from tensorboardX import SummaryWriter
from random_erasing import RandomErasing
import build_transforms

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda_1', default=0.7, type=float, help='weights for tca')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = './Visualization/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = './Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
elif dataset == 'llcm':
    data_path = './dataset/LLCM/'
    log_path = args.log_path + 'llcm_log/'
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
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

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)


gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(n_class, arch=args.arch)
net.to(device)
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

# define loss function
criterion_id = nn.CrossEntropyLoss()
loader_batch = args.batch_size * args.num_pos
criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_tca = TcaLoss(margin=0.7)

criterion_id.to(device)
criterion_tri.to(device)
criterion_tca.to(device)

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
    ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 60:
        lr = args.lr * 0.1
    elif epoch >= 60 and epoch < 90:
        lr = args.lr * 0.01
    elif epoch >= 90:
        lr = args.lr * 0.001

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr

def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    tca_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input10, input11, input20, input21,  label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label1, label2, label2), 0)

        input10 = Variable(input10.cuda())
        input11 = Variable(input11.cuda())
        input20 = Variable(input20.cuda())
        input21 = Variable(input21.cuda())
        labels = Variable(labels.cuda())

        data_time.update(time.time() - end)


        feat, out = net(input10, input11, input20, input21)

        loss_id = criterion_id(out, labels.long())
        
        lbs = torch.cat((label1, label2), 0)

        feat1, feat2, feat3, feat4 = torch.chunk(feat, 4, 0)

        loss_tri = (criterion_tri(torch.cat((feat1, feat3), 0), lbs)[0] + criterion_tri(torch.cat((feat1, feat4), 0), lbs)[0] + criterion_tri(torch.cat((feat2, feat3), 0), lbs)[0] + criterion_tri(torch.cat((feat2, feat4), 0), lbs)[0])/4

        loss_tca = criterion_tca(feat, labels) * args.lambda_1

        _, predicted = out.max(1)
        correct += (predicted.eq(labels).sum().item())

        loss = loss_id + loss_tri + loss_tca
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input10.size(0))
        id_loss.update(loss_id.item(), 2 * input10.size(0))
        tri_loss.update(loss_tri.item(), 2 * input10.size(0))
        tca_loss.update(loss_tca.item(), 2 * input10.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'TcaLoss: {tca_loss.val:.4f} ({tca_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,tca_loss=tca_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)

def test(epoch):
    # switch to evaluation mode
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
            feat1, feat2= torch.chunk(feat, 2, 0)
            feat_att1, feat_att2= torch.chunk(feat_att, 2, 0)
            gall_feat1[ptr:ptr + batch_num, :] = feat1.detach().cpu().numpy()
            gall_feat2[ptr:ptr + batch_num, :] = feat2.detach().cpu().numpy()
            gall_feat_att1[ptr:ptr + batch_num, :] = feat_att1.detach().cpu().numpy()
            gall_feat_att2[ptr:ptr + batch_num, :] = feat_att2.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
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
            feat_att1, feat_att2= torch.chunk(feat_att, 2, 0)
            query_feat1[ptr:ptr + batch_num, :] = feat1.detach().cpu().numpy()
            query_feat2[ptr:ptr + batch_num, :] = feat2.detach().cpu().numpy()
            query_feat_att1[ptr:ptr + batch_num, :] = feat_att1.detach().cpu().numpy()
            query_feat_att2[ptr:ptr + batch_num, :] = feat_att2.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
    distmat2 = np.matmul(query_feat1, np.transpose(gall_feat2))
    distmat3 = np.matmul(query_feat2, np.transpose(gall_feat1))
    distmat4 = np.matmul(query_feat2, np.transpose(gall_feat2))


    distmat_att1 = np.matmul(query_feat_att1, np.transpose(gall_feat_att1))
    distmat_att2 = np.matmul(query_feat_att1, np.transpose(gall_feat_att2))
    distmat_att3 = np.matmul(query_feat_att2, np.transpose(gall_feat_att1))
    distmat_att4 = np.matmul(query_feat_att2, np.transpose(gall_feat_att2))

    distmat_total = distmat1 + distmat2 + distmat3 + distmat4
    distmat_total_att = distmat_att1 + distmat_att2 + distmat_att3 + distmat_att4
    distmat_all = distmat1 + distmat_att1 + distmat2 + distmat_att2 + distmat3 + distmat_att3 + distmat4 + distmat_att4

    # evaluation
    if dataset == 'regdb':
        cmc_att1, mAP_att1, mINP_att1 = eval_regdb(-distmat_att1, query_label, gall_label)
        cmc_att2, mAP_att2, mINP_att2 = eval_regdb(-distmat_att2, query_label, gall_label)
        cmc_att3, mAP_att3, mINP_att3 = eval_regdb(-distmat_att3, query_label, gall_label)
        cmc_att4, mAP_att4, mINP_att4 = eval_regdb(-distmat_att4, query_label, gall_label)
        cmc_att5, mAP_att5, mINP_att5 = eval_regdb(-distmat_total_att, query_label, gall_label)
        cmc, mAP, mINP = eval_regdb(-distmat_all, query_label, gall_label)
    elif dataset == 'sysu':
        cmc_att1, mAP_att1, mINP_att1 = eval_sysu(-distmat_att1, query_label, gall_label, query_cam, gall_cam)
        cmc_att2, mAP_att2, mINP_att2 = eval_sysu(-distmat_att2, query_label, gall_label, query_cam, gall_cam)
        cmc_att3, mAP_att3, mINP_att3 = eval_sysu(-distmat_att3, query_label, gall_label, query_cam, gall_cam)
        cmc_att4, mAP_att4, mINP_att4 = eval_sysu(-distmat_att4, query_label, gall_label, query_cam, gall_cam)
        cmc_att5, mAP_att5, mINP_att5 = eval_sysu(-distmat_total_att, query_label, gall_label, query_cam, gall_cam)
        cmc, mAP, mINP = eval_sysu(-distmat_all, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'llcm':
        cmc_att1, mAP_att1, mINP_att1 = eval_llcm(-distmat_att1, query_label, gall_label, query_cam, gall_cam)
        cmc_att2, mAP_att2, mINP_att2 = eval_llcm(-distmat_att2, query_label, gall_label, query_cam, gall_cam)
        cmc_att3, mAP_att3, mINP_att3 = eval_llcm(-distmat_att3, query_label, gall_label, query_cam, gall_cam)
        cmc_att4, mAP_att4, mINP_att4 = eval_llcm(-distmat_att4, query_label, gall_label, query_cam, gall_cam)
        cmc_att5, mAP_att5, mINP_att5 = eval_llcm(-distmat_total_att, query_label, gall_label, query_cam, gall_cam)
        cmc, mAP, mINP = eval_llcm(-distmat_all, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    return cmc_att1, mAP_att1, mINP_att1, \
           cmc_att2, mAP_att2, mINP_att2, \
           cmc_att3, mAP_att3, mINP_att3, \
           cmc_att4, mAP_att4, mINP_att4, \
           cmc_att5, mAP_att5, mINP_att5, \
           cmc, mAP, mINP

# training
print('==> Start Training...')
for epoch in range(start_epoch, 121 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch > 0 and epoch % 1 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc_att1, mAP_att1, mINP_att1, \
        cmc_att2, mAP_att2, mINP_att2, \
        cmc_att3, mAP_att3, mINP_att3, \
        cmc_att4, mAP_att4, mINP_att4, \
        cmc_att5, mAP_att5, mINP_att5, \
        cmc, mAP, mINP= test(epoch)
        # save model
        if cmc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'mINP': mINP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        print(
            'VIS-IR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att1[0], cmc_att1[4], cmc_att1[9], cmc_att1[19], mAP_att1, mINP_att1))
        print(
            'VIS-MR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att2[0], cmc_att2[4], cmc_att2[9], cmc_att2[19], mAP_att2, mINP_att2))
        print(
            'MIS-IR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att3[0], cmc_att3[4], cmc_att3[9], cmc_att3[19], mAP_att3, mINP_att3))
        print(
            'MIS-MR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att4[0], cmc_att4[4], cmc_att4[9], cmc_att4[19], mAP_att4, mINP_att4))
        print(
            'ALL FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att5[0], cmc_att5[4], cmc_att5[9], cmc_att5[19], mAP_att5, mINP_att5))
        print(
            'ALL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('Best Epoch [{}]'.format(best_epoch))
