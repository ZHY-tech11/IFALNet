import scipy.io
import torch
import numpy as np
#import time
import os
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, ticker
from matplotlib.ticker import PercentFormatter
import scipy.io
import argparse

parser = argparse.ArgumentParser(description='IFALNet intra-class and inter-class distances')
parser.add_argument('--mode', default='all', type=str, help='VIS-IR or VIS-ItM or VtM-IR or VtM-ItM')
args = parser.parse_args()


######################################################################
result = scipy.io.loadmat('tsne.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = torch.FloatTensor(result['query_label'][0])
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = torch.FloatTensor(result['gallery_label'][0])


gallery_feature1, gallery_feature2 = torch.chunk(gallery_feature, 2, 0)
gallery_label1, gallery_label2 = torch.chunk(gallery_label, 2, 0)

query_feature1, query_feature2 = torch.chunk(query_feature, 2, 0)
query_label1, query_label2 = torch.chunk(query_label, 2, 0)

'''
if args.mode == 'VIS-IR':
    gallery_feature = gallery_feature1
    query_feature = query_feature1

if args.mode == 'VIS-ItM':
    gallery_feature = gallery_feature1
    query_feature = query_feature2

if args.mode == 'VtM-IR':
    gallery_feature = gallery_feature2
    query_feature = query_feature1

if args.mode == 'VtM-ItM':
    gallery_feature = gallery_feature2
    query_feature = query_feature2


gallery_label = gallery_label1
query_label = query_label1
'''

query_feature = query_feature.detach().cpu().numpy()
gallery_feature = gallery_feature.detach().cpu().numpy()


mask = query_label.expand(len(gallery_label), len(query_label)).eq(gallery_label.expand(len(query_label), len(gallery_label)).t()).cuda()

distmat = torch.FloatTensor(1 - np.matmul(gallery_feature, np.transpose(query_feature))).cuda() #Cosine distance
intra = distmat[mask]
inter = distmat[mask == 0]

######################################################################
plt.rcParams.update({'font.size': 14})


fig, ax = plt.subplots()
b = np.linspace(0.3, 1.3, num=1000)

ax.hist(intra.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color='#4169E1', density=True, label='Intra-class')
ax.hist(inter.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color='#008080', density=True, label='Inter-class')

ax.set_ylim(0, 7)
#ax.set_xlabel('(a) Baseline Distance', fontweight='bold', fontsize=16)
ax.set_ylabel('Frequency', fontweight='bold')
ax.legend(loc='upper right', fontsize='small')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x)))


fig.savefig('scatter.svg',dpi=1000,format='svg')
plt.show()
