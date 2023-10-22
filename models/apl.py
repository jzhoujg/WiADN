import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

import scipy.io as sio

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class MaskGenerator(nn.Module):

    def __init__(self, num_channel_1, num_channel_2, isfirstblock=False):
        super(MaskGenerator, self).__init__()

        self.isisfirstblock = isfirstblock
        self.total_channel = num_channel_1 + num_channel_2
        self.maskgenerator_1 = nn.Sequential(
                                    nn.Conv1d(in_channels=self.total_channel, out_channels=self.total_channel, kernel_size= 1, stride=1, padding=0),
                                    nn.BatchNorm1d(self.total_channel),
                                    nn.ReLU()
        )

        self.maskgenerator_2 = nn.Sequential(
            nn.Conv1d(in_channels=self.total_channel, out_channels=self.total_channel, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm1d(self.total_channel),
            nn.Sigmoid()
        )



    def forward(self, input_basic, input_last_att):

        if self.isisfirstblock:
            identity = input_basic
        else:
            identity = torch.cat((input_basic,input_last_att),1)
        x = identity
        x = self.maskgenerator_1(x)
        x = self.maskgenerator_2(x)
        out = torch.mul(x,identity)

        return out

class FeatureExtractor(nn.Module):

    def __init__(self, in_channels, out_channels, downsampling = None):
        super(FeatureExtractor, self).__init__()

        self.downsampling = downsampling
        self.extrator_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )


    def forward(self, x):

        out = self.extrator_1(x)
        if self.downsampling:
            out = self.downsampling(out)

        return out


class AttentionModule(nn.Module):

    def __init__(self,num_channel_1, num_channel_2, in_channels, out_channels, isfirstblock=False, downsampling=None):
        super(AttentionModule, self).__init__()
        self.MaskGenerator = MaskGenerator(num_channel_1=num_channel_1,num_channel_2=num_channel_2,isfirstblock=isfirstblock)
        self.FeatureExtractor = FeatureExtractor(in_channels=in_channels,out_channels=out_channels,downsampling=downsampling)


    def forword(self,input_basic, input_last_att):

        x = self.MaskGenerator.forward(input_basic,input_last_att)
        out = self.FeatureExtractor.forward(x)


        return out


# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.conv2 = conv3x3(planes, planes, stride)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm1d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out


class ResNet(nn.Module):

    def __init__(self, block, layers,  inchannel=52, activity_num=7, location_num=6):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.att1 = AttentionModule(128, 0, 128, 128, isfirstblock=True)



        self.downsampling1 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.att2 = AttentionModule(128, 128, 256, 128, isfirstblock=False,downsampling=self.downsampling1)


        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.downsampling2 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.att3 = AttentionModule(128, 128, 256, 256, isfirstblock=False, downsampling=self.downsampling2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.downsampling3 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.att4 = AttentionModule(256, 256, 512, 512, isfirstblock=False, downsampling=self.downsampling3)

        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, activity_num)

        self.LOCClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )


        self.loc_fc = nn.Linear(512 * block.expansion, location_num)
        self.loc_fc_f = nn.Linear(256, location_num)

        #
        # self.fc1 = nn.Linear(512 * block.expansion, )
        # self.fc2 = nn.Linear(512 * block.expansion, )
        # self.fc3 = nn.Linear(512 * block.expansion, )
        # self.fc4 = nn.Linear(512 * block.expansion, )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #
        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y


    def forward(self, x):
        # sampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # feature extractor
        s1 = self.att1.forword(input_basic=x,input_last_att=None)
        c1 = self.layer1(x)

        s2 = self.att2.forword(input_basic=c1,input_last_att=s1)
        c2 = self.layer2(c1)


        s3 = self.att3.forword(input_basic=c2,input_last_att=s2)
        c3 = self.layer3(c2)

        s4 = self.att4.forword(input_basic=c3, input_last_att=s3)
        c4 = self.layer4(c3)

        #

        act = self.ACTClassifier(s4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        loc = self.LOCClassifier(c4)
        loc = loc.view(loc.size(0), -1)
        loc1 = self.loc_fc(loc)


        return act1, loc1, x, c1, c2, c3, c4, act, loc


if __name__ == '__main__':
    # load data
    data_amp = sio.loadmat('train_data_split_amp.mat')
    train_data_amp = data_amp['train_data']
    train_data = train_data_amp
    num_train_instances = len(train_data)
    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)

    # train_data = train_data.view(num_train_instances, 1, -1)
    # train_label = train_label.view(num_train_instances, 2)
    data = torch.zeros(1,52,192)
    data[0,:,:] = train_data[0,:,:]
    aplnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=52)
    out = aplnet(data)




