import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        _, top_3 = y.topk(3, dim=1)
        return x * y.expand_as(x), top_3


class visible_moduleA(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_moduleA, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v
        # self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.visible.conv1(x)
        # x = self.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x

class thermal_moduleA(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_moduleA, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        # self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.thermal.conv1(x)
        # x = self.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.thermal_moduleA = thermal_moduleA(arch=arch)
        self.visible_moduleA = visible_moduleA(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.encode1 = nn.Conv2d(3, 1, 1)
        self.encode1.apply(my_weights_init)
        self.fc1 = nn.Conv2d(1, 1, 1)
        self.fc1.apply(my_weights_init)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn1.apply(weights_init_kaiming)

        self.encode2 = nn.Conv2d(3, 1, 1)
        self.encode2.apply(my_weights_init)
        self.fc2 = nn.Conv2d(1, 1, 1)
        self.fc2.apply(my_weights_init)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn2.apply(weights_init_kaiming)

        self.encode3 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=1)
        self.encode3.apply(my_weights_init)

        self.SEnet = SELayer(channel=6)

        self.decode = nn.Conv2d(3, 3, 1)
        self.decode.apply((my_weights_init))

        self.MIA1 = MIA(channel=256)
        self.MIA2 = MIA(channel=512)
        self.MIA3 = MIA(channel=1024)

        self.MFA = MFA(in_channels=1024)

    def forward(self, x1_1, x1_2, x2_1, x2_2, modal=0):
        if modal == 0:
            x1_2 = F.relu(self.encode1(x1_2))
            x1_2 = self.bn1(F.relu(self.fc1(x1_2)))

            x2_2 = F.relu(self.encode2(x2_2))
            x2_2 = self.bn2(F.relu(self.fc2(x2_2)))

            x = F.relu(self.encode3(torch.cat((x1_2, x2_2), 0)))
            x, k = self.SEnet(x)

            list = []
            for i in range(x.size(0)):
                n1 = x[i:i + 1, int(k[i][0]):int(k[i][0]) + 1, :, :]
                n2 = x[i:i + 1, int(k[i][1]):int(k[i][1]) + 1, :, :]
                n3 = x[i:i + 1, int(k[i][2]):int(k[i][2]) + 1, :, :]
                xx = torch.cat((n1, n2, n3), 1)
                list.append(xx)
            x = torch.cat(list, 0)
            x = F.relu(self.decode(x))

            x1_2, x2_2 = torch.chunk(x, 2, 0)

            x1_1 = self.visible_module(x1_1)
            x1_2 = self.visible_moduleA(x1_2)
            x2_1 = self.thermal_module(x2_1)
            x2_2 = self.thermal_moduleA(x2_2)

            x = torch.cat((x1_1, x1_2, x2_1, x2_2), 0)

        elif modal == 1:
            x1_2 = F.relu(self.encode1(x1_2))
            x1_2 = self.bn1(F.relu(self.fc1(x1_2)))
            x1_2 = F.relu(self.encode3(x1_2))
            x1_2, k = self.SEnet(x1_2)
            list = []
            for i in range(x1_2.size(0)):
                n1 = x1_2[i:i + 1, int(k[i][0]):int(k[i][0]) + 1, :, :]
                n2 = x1_2[i:i + 1, int(k[i][1]):int(k[i][1]) + 1, :, :]
                n3 = x1_2[i:i + 1, int(k[i][2]):int(k[i][2]) + 1, :, :]
                xx = torch.cat((n1, n2, n3), 1)
                list.append(xx)
            x1_2 = torch.cat(list, 0)

            x1_2 = F.relu(self.decode(x1_2))

            x1_1 = self.visible_module(x1_1)
            x1_2 = self.visible_moduleA(x1_2)
            x = torch.cat((x1_1, x1_2), 0)

        elif modal == 2:
            x2_2 = F.relu(self.encode2(x2_2))
            x2_2 = self.bn2(F.relu(self.fc2(x2_2)))
            x2_2 = F.relu(self.encode3(x2_2))
            x2_2, k = self.SEnet(x2_2)

            list = []
            for i in range(x2_2.size(0)):
                n1 = x2_2[i:i + 1, int(k[i][0]):int(k[i][0]) + 1, :, :]
                n2 = x2_2[i:i + 1, int(k[i][1]):int(k[i][1]) + 1, :, :]
                n3 = x2_2[i:i + 1, int(k[i][2]):int(k[i][2]) + 1, :, :]
                xx = torch.cat((n1, n2, n3), 1)
                list.append(xx)
            x2_2 = torch.cat(list, 0)

            x2_2 = F.relu(self.decode(x2_2))

            x2_1 = self.thermal_module(x2_1)
            x2_2 = self.thermal_moduleA(x2_2)
            x = torch.cat((x2_1, x2_2), 0)

            # shared block
        x_res1, x_res2 = 0, 0
        x = self.base_resnet.base.layer1(x)
        x_1, x_2 = chunk_feature(x, modal=modal)
        x_1, x_2, x_res1, x_res2 = self.MIA1(x_1, x_2, x_res1, x_res2, layer=1)
        x = concat_feature(x_1, x_2, modal=modal)

        x = self.base_resnet.base.layer2(x)
        x_1, x_2 = chunk_feature(x, modal=modal)
        x_1, x_2, x_res1, x_res2 = self.MIA2(x_1, x_2, x_res1, x_res2, layer=2)
        x = concat_feature(x_1, x_2, modal=modal)

        x = self.base_resnet.base.layer3(x)
        x_1, x_2 = chunk_feature(x, modal=modal)
        x_1, x_2, _, _ = self.MIA3(x_1, x_2, x_res1, x_res2, layer=3)
        x = concat_feature(x_1, x_2, modal=modal)
        x = self.MFA(x)

        x = self.base_resnet.base.layer4(x)

        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

def chunk_feature(x, modal):
    if modal == 0:
        x1_1, x1_2, x2_1, x2_2 = torch.chunk(x, 4, 0)
        x_1, x_2 = torch.cat((x1_1, x2_1), 0), torch.cat((x1_2, x2_2), 0)
        return x_1, x_2
    elif modal == 1 or modal == 2:
        x_1, x_2 = torch.chunk(x, 2, 0)
        return x_1, x_2

def concat_feature(x_1, x_2, modal):
    if modal == 0:
        x1, x3 = torch.chunk(x_1, 2, 0)
        x2, x4 = torch.chunk(x_2, 2, 0)
        x = torch.cat((x1, x2, x3, x4), 0)
        return x
    elif modal == 1 or modal == 2:
        x = torch.cat((x_1, x_2), 0)
        return x

class MFA(nn.Module):
    def __init__(self, in_channels, BatchNorm=nn.BatchNorm2d):
        super(MFA, self).__init__()
        self.FC11 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, stride=1, padding=5, bias=False, dilation=5)
        self.FC13.apply(weights_init_kaiming)
        self.FC14 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, stride=1, padding=7, bias=False, dilation=7)
        self.FC14.apply(weights_init_kaiming)

        self.FC21 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC21.apply(weights_init_kaiming)
        self.FC22 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, stride=1, padding=4, bias=False, dilation=4)
        self.FC22.apply(weights_init_kaiming)
        self.FC23 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, stride=1, padding=6, bias=False, dilation=6)
        self.FC23.apply(weights_init_kaiming)
        self.FC24 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, stride=1, padding=8, bias=False, dilation=8)
        self.FC24.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(in_channels//4, in_channels//2, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.01)
        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels,
                      kernel_size=1, bias=False),
            BatchNorm(in_channels)
        )

    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x) + self.FC14(x)) / 4
        x1 = self.dropout(self.FC1(F.relu(x1)))
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x) + self.FC24(x)) / 4
        x2 = self.dropout(self.FC1(F.relu(x2)))

        x = self.FC2(x)
        att = torch.sigmoid(x)
        x = self.conv_o((1 - att) * x1 + x + att * x2)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x1, x2):
        out1 = x1 * self.ca(x1)
        out2 = x2 * self.ca(x2)

        result1 = out1 * self.sa(out1)
        result2 = out2 * self.sa(out2)
        return result1, result2

class MIA(nn.Module):
    def __init__(self, channel):
        super(MIA, self).__init__()
        self.channel = channel

        self.CBAM_ori = CBAM(self.channel)

        self.conv_narrow1 = nn.Conv2d(self.channel, self.channel * 2, kernel_size=2, stride=2, padding=1, dilation=2)
        self.conv_narrow2 = nn.Conv2d(self.channel, self.channel * 2, kernel_size=2, stride=2, padding=1, dilation=2)

    def forward(self, x_1, x_2, x_res1, x_res2, layer):
        if layer == 1:
            x_1, x_2 = self.CBAM_ori(x_1, x_2)

            x_res1 = self.conv_narrow1(x_1)
            x_res2 = self.conv_narrow2(x_2)

            return x_1, x_2, x_res1, x_res2

        elif layer == 2:
            x_1 = x_1 + 0.2 * x_res1
            x_2 = x_2 + 0.2 * x_res2

            x_1, x_2 = self.CBAM_ori(x_1, x_2)

            x_res1 = x_1
            x_res2 = x_2
            x_res1 = self.conv_narrow1(x_res1)
            x_res2 = self.conv_narrow2(x_res2)

            return x_1, x_2, x_res1, x_res2

        elif layer == 3:
            x_1 = x_1 + 0.2 * x_res1
            x_2 = x_2 + 0.2 * x_res2

            x_1, x_2 = self.CBAM_ori(x_1, x_2)

            return x_1, x_2, x_res1, x_res2


