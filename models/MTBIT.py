import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import models 

import functools
from einops import rearrange

import models.resnet as rn
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d

from fightingcv_attention.attention.CBAM import CBAMBlock,CBAMECABlock
from fightingcv_attention.attention.SKAttention import SKAttention
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.ECAAttention import ECAAttention

class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True, learnable = False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = rn.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
           
        elif backbone == 'resnet34':
            self.resnet = rn.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = rn.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        elif backbone == 'dino:resnet50':
            self.resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            expand = 4
        elif backbone == 'encoder:resnet50':
            self.resnet = models.resnet50(pretrained = True)
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()


        self.upsamplex2 = nn.Upsample(scale_factor=2)
        
        self.learnable = learnable 
        
        self.upsamplex2l1_single = nn.ConvTranspose2d(256*expand, 256*expand, 4, 2, 1)
        
        if self.learnable:
            self.upsamplex2l1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
            self.upsamplex2l2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.regressor = TwoLayerConv2d(in_channels=32, out_channels=1)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.active3d = nn.Tanh()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1_single(x_8)
            else:
                x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x

class ResNet_FPN5(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True, learnable = False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_FPN5, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = rn.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
           
        elif backbone == 'resnet34':
            self.resnet = rn.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = rn.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        elif backbone == 'dino:resnet50':
            self.resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            expand = 4
        elif backbone == 'encoder:resnet50':
            self.resnet = models.resnet50(pretrained = True)
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()


        self.upsamplex2 = nn.Upsample(scale_factor=2)
        
        self.learnable = learnable 
        
        self.upsamplex2l1_single = nn.ConvTranspose2d(256*expand, 256*expand, 4, 2, 1)
        
        if self.learnable:
            self.upsamplex2l1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
            self.upsamplex2l2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.regressor = TwoLayerConv2d(in_channels=32, out_channels=1)

        self.resnet_stages_num = resnet_stages_num

        # self.fpn = FPN_S4()  #####satge = 4
        self.fpn2 = FPN_S5(out_channels=512)  #####satge = 5
        # self.fpn3 = FPN_S3(out_channels=128)  #####satge = 3
        # self.fpn4 = FPN_S2(out_channels=64)  #####satge = 2

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        ############ my     
        elif self.resnet_stages_num == 2:
            layers = 64
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.active3d = nn.Tanh()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        xout = [x]
        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        xout.append(x_4)
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        xout.append(x_8)

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
            xout.append(x_8)

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            xout.append(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1_single(x_8)
            else:
                x = self.upsamplex2(x_8)
        else:
            x = x_8
        
      
        x = self.fpn2(xout)  ############五层特征融合
        # x = self.fpn(xout)  ############四层特征融合
        # output layers
        x = self.conv_pred(x)
        return x

class ResNet_FPN4(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True, learnable = False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_FPN4, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = rn.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
           
        elif backbone == 'resnet34':
            self.resnet = rn.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = rn.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        elif backbone == 'dino:resnet50':
            self.resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            expand = 4
        elif backbone == 'encoder:resnet50':
            self.resnet = models.resnet50(pretrained = True)
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()


        self.upsamplex2 = nn.Upsample(scale_factor=2)
        
        self.learnable = learnable 
        
        self.upsamplex2l1_single = nn.ConvTranspose2d(256*expand, 256*expand, 4, 2, 1)
        
        if self.learnable:
            self.upsamplex2l1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
            self.upsamplex2l2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.regressor = TwoLayerConv2d(in_channels=32, out_channels=1)

        self.resnet_stages_num = resnet_stages_num

        self.fpn = FPN_S4()  #####satge = 4
        # self.fpn2 = FPN_S5(out_channels=512)  #####satge = 5
        # self.fpn3 = FPN_S3(out_channels=128)  #####satge = 3
        # self.fpn4 = FPN_S2(out_channels=64)  #####satge = 2

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        ############ my     
        elif self.resnet_stages_num == 2:
            layers = 64
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.active3d = nn.Tanh()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        xout = [x]
        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        xout.append(x_4)
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        xout.append(x_8)

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
            xout.append(x_8)

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            xout.append(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1_single(x_8)
            else:
                x = self.upsamplex2(x_8)
        else:
            x = x_8
        
      
        # x = self.fpn2(xout)  ############五层特征融合
        x = self.fpn(xout)  ############四层特征融合
        # output layers
        x = self.conv_pred(x)
        return x

class ResNet_FPN3(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True, learnable = False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_FPN3, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = rn.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
           
        elif backbone == 'resnet34':
            self.resnet = rn.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = rn.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        elif backbone == 'dino:resnet50':
            self.resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            expand = 4
        elif backbone == 'encoder:resnet50':
            self.resnet = models.resnet50(pretrained = True)
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()


        self.upsamplex2 = nn.Upsample(scale_factor=2)
        
        self.learnable = learnable 
        
        self.upsamplex2l1_single = nn.ConvTranspose2d(256*expand, 256*expand, 4, 2, 1)
        
        if self.learnable:
            self.upsamplex2l1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
            self.upsamplex2l2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.regressor = TwoLayerConv2d(in_channels=32, out_channels=1)

        self.resnet_stages_num = resnet_stages_num

        # self.fpn = FPN_S4()  #####satge = 4
        # self.fpn2 = FPN_S5(out_channels=512)  #####satge = 5
        self.fpn3 = FPN_S3(out_channels=128)  #####satge = 3
        # self.fpn4 = FPN_S2(out_channels=64)  #####satge = 2

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        ############ my     
        elif self.resnet_stages_num == 2:
            layers = 64
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.active3d = nn.Tanh()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        xout = [x]
        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        xout.append(x_4)
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        xout.append(x_8)

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
            xout.append(x_8)

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            xout.append(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1_single(x_8)
            else:
                x = self.upsamplex2(x_8)
        else:
            x = x_8
        
      
        # x = self.fpn2(xout)  ############五层特征融合
        # x = self.fpn(xout)  ############四层特征融合
        x = self.fpn3(xout)  ############三层特征融合
        # output layers
        x = self.conv_pred(x)
        return x

class ResNet_FPN2(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True, learnable = False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_FPN2, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = rn.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
           
        elif backbone == 'resnet34':
            self.resnet = rn.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = rn.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        elif backbone == 'dino:resnet50':
            self.resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            expand = 4
        elif backbone == 'encoder:resnet50':
            self.resnet = models.resnet50(pretrained = True)
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()


        self.upsamplex2 = nn.Upsample(scale_factor=2)
        
        self.learnable = learnable 
        
        self.upsamplex2l1_single = nn.ConvTranspose2d(256*expand, 256*expand, 4, 2, 1)
        
        if self.learnable:
            self.upsamplex2l1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
            self.upsamplex2l2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.regressor = TwoLayerConv2d(in_channels=32, out_channels=1)

        self.resnet_stages_num = resnet_stages_num

        # self.fpn = FPN_S4()  #####satge = 4
        # self.fpn2 = FPN_S5(out_channels=512)  #####satge = 5
        # self.fpn3 = FPN_S3(out_channels=128)  #####satge = 3
        self.fpn4 = FPN_S2(out_channels=64)  #####satge = 2

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        ############ my     
        elif self.resnet_stages_num == 2:
            layers = 64
    
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.active3d = nn.Tanh()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        xout = [x]
        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        xout.append(x_4)
        # x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        # xout.append(x_8)

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
            xout.append(x_8)

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            xout.append(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        # if self.if_upsample_2x:
        #     if self.learnable:
        #         x = self.upsamplex2l1_single(x_8)
        #     else:
        #         x = self.upsamplex2(x_8)
        # else:
        #     x = x_8
      
        # x = self.fpn2(xout)  ############五层特征融合
        # x = self.fpn(xout)  ############四层特征融合
        # x = self.fpn3(xout)  ############三层特征融合
        x = self.fpn4(xout)  ############两层特征融合
       
        # output layers
        x = self.conv_pred(x)
        return x

class MTBIT(ResNet):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)
        # feature differencing
        x = x2 - x1
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)
        x3d = self.active3d(x3d)

        return x2d, x3d

class MTBIT_FPN5(ResNet_FPN5):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT_FPN5, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        # self.diff = Diff(in_channels= 2* dim, out_channels=dim)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)

        # feature differencing
        x = x2 - x1
        # x = self.diff(torch.cat((x1, x2), dim=1))
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)  
        x3d = self.active3d(x3d)

        return x2d, x3d

class MTBIT_FPN5_DIFF(ResNet_FPN5):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT_FPN5_DIFF, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        self.diff = Diff(in_channels= 2* dim, out_channels=dim)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)

        # feature differencing
        # x = x2 - x1
        x = self.diff(torch.cat((x1, x2), dim=1))
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)  
        x3d = self.active3d(x3d)

        return x2d, x3d

class MTBIT_FPN4(ResNet_FPN4):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT_FPN4, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        # self.diff = Diff(in_channels= 2* dim, out_channels=dim)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)

        # feature differencing
        x = x2 - x1
        # x = self.diff(torch.cat((x1, x2), dim=1))
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)  
        x3d = self.active3d(x3d)

        return x2d, x3d

class MTBIT_FPN4_DIFF(ResNet_FPN4):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT_FPN4_DIFF, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        self.diff = Diff(in_channels= 2* dim, out_channels=dim)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)

        # feature differencing
        # x = x2 - x1
        x = self.diff(torch.cat((x1, x2), dim=1))
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)  
        x3d = self.active3d(x3d)

        return x2d, x3d

class MTBIT_FPN4_abs(ResNet_FPN4):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT_FPN4_abs, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        # self.diff = Diff(in_channels= 2* dim, out_channels=dim)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)

        # feature differencing
        # x = x2 - x1
        # x = self.diff(torch.cat((x1, x2), dim=1))
        x = torch.abs(x2 - x1)
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)  
        x3d = self.active3d(x3d)

        return x2d, x3d

class MTBIT_FPN3(ResNet_FPN3):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT_FPN3, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        # self.diff = Diff(in_channels= 2* dim, out_channels=dim)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)

        # feature differencing
        x = x2 - x1
        # x = self.diff(torch.cat((x1, x2), dim=1))
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)  
        x3d = self.active3d(x3d)

        return x2d, x3d

class MTBIT_FPN2(ResNet_FPN2):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT_FPN2, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        # self.diff = Diff(in_channels= 2* dim, out_channels=dim)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)

        # feature differencing
        x = x2 - x1
        # x = self.diff(torch.cat((x1, x2), dim=1))
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)  
        x3d = self.active3d(x3d)

        return x2d, x3d

class MTBIT_FPN4_ATTEN(ResNet_FPN4):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT_FPN4_ATTEN, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        # self.diff = Diff(in_channels= 2* dim, out_channels=dim)

        ###########SK
        # self.sk = SKAttention(channel=32,reduction=4)
        # self.se = SEAttention(channel=32,reduction=4)
        # self.cbam = CBAMBlock(channel=32,reduction=4,kernel_size=7)
        # self.cbam_eca = CBAMECABlock(kernel_size_ca=3, kernel_size_sa=7)
        self.eca = ECAAttention(kernel_size=3)
        

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)

        # # #############CBAM
        # x1 = self.cbam(x1)
        # x2 = self.cbam(x2)

        # # #############cbameca
        # x1 = self.cbam_eca(x1)
        # x2 = self.cbam_eca(x2)

        # #############sk
        # x1 = self.sk(x1)
        # x2 = self.sk(x2)

        # # #############se
        # x1 = self.se(x1)
        # x2 = self.se(x2)

        #############eca
        x1 = self.eca(x1)
        x2 = self.eca(x2)
        

        # feature differencing
        x = x2 - x1
        # x = self.diff(torch.cat((x1, x2), dim=1))
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)  
        x3d = self.active3d(x3d)

        return x2d, x3d

class MTBIT_FPN5_ATTEN(ResNet_FPN5):
    def __init__(self, input_nc, output_nc, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 if_upsample_2x=True,
                 backbone='resnet18', learnable = False,
                 decoder_softmax=True, 
                 ):
                     
        super(MTBIT_FPN5_ATTEN, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, 
                                               learnable = learnable,)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,padding=0, bias=False)
        self.learnable = learnable 
        self.token_trans = token_trans
        dim = 32
        mlp_dim = 2*dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        # decoder_pos_size = 256//4
        decoder_pos_size = 200//4
        # decoder_pos_size = 100//4 
        self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        # self.diff = Diff(in_channels= 2* dim, out_channels=dim)

        ###########SK
        # self.sk = SKAttention(channel=32,reduction=4)
        # self.se = SEAttention(channel=32,reduction=4)
        # self.cbam = CBAMBlock(channel=32,reduction=4,kernel_size=7)
        # self.cbam_eca = CBAMECABlock(kernel_size_ca=3, kernel_size_sa=7)
        self.eca = ECAAttention(kernel_size=3)
        

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer(self, x):
        # if self.with_pos:
        x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)

        # # #############CBAM
        # x1 = self.cbam(x1)
        # x2 = self.cbam(x2)

        # # #############cbameca
        # x1 = self.cbam_eca(x1)
        # x2 = self.cbam_eca(x2)

        # #############sk
        # x1 = self.sk(x1)
        # x2 = self.sk(x2)

        # # #############se
        # x1 = self.se(x1)
        # x2 = self.se(x2)

        #############eca
        x1 = self.eca(x1)
        x2 = self.eca(x2)
        

        # feature differencing
        x = x2 - x1
        # x = self.diff(torch.cat((x1, x2), dim=1))
        
        if not self.if_upsample_2x:
            if self.learnable:
                x = self.upsamplex2l1(x)
                x = self.upsamplex2l2(x)
            else:
                x = self.upsamplex4(x)
        
        if self.learnable:
            x = self.upsamplex2l1(x)
            x = self.upsamplex2l2(x)
        else:
            x = self.upsamplex4(x)
        
        # forward small cnn
        x2d = self.classifier(x)
        if self.output_sigmoid:
            x2d = self.sigmoid(x2d)

        x3d = self.regressor(x)  
        x3d = self.active3d(x3d)

        return x2d, x3d

class FPN_S4(nn.Module):
    """resnet(S4) + FPN

    return: 融合后的特征层
    """
    def __init__(self, in_channels = [64, 64, 128, 256], out_channels = 256):
        super().__init__()

        self.conv1by1_4 = nn.Conv2d(in_channels[-1], out_channels, 1)
        self.conv1by1_3 = nn.Conv2d(in_channels[-2], out_channels, 1)
        self.conv1by1_2 = nn.Conv2d(in_channels[-3], out_channels, 1)
        self.conv1by1_1 = nn.Conv2d(in_channels[-4], out_channels, 1)


    def forward(self, x):
        f4 = x[-1]  #####[16,256,32,32]  (no upsampling)
        f3 = x[-2]  #####[16,128,32,32]
        f2 = x[-3]  #####[16,64,64,64]
        f1 = x[-4]  #####[16,64,64,64]
    

        # f4 = self.conv1by1_4(f4) 
        f4 = F.interpolate(f4, scale_factor= 2 , mode="nearest")  #####[16,256,64,64]  
        
        f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")  #####[16,128,64,64]
        f3 = self.conv1by1_3(f3) + f4                             #####[16,256,64,64]  
        # f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")

        f2 = self.conv1by1_2(f2) + f3
        # f2 = F.interpolate(f2, scale_factor= 2 , mode="nearest")

        f1 = self.conv1by1_1(f1) + f2
       
        return f1

class FPN_S5(nn.Module):
    """resnet(S5) + FPN

    return: 融合后的特征层
    """
    def __init__(self, in_channels = [64, 64, 128, 256, 512], out_channels = 256):
        super().__init__()

        self.conv1by1_5 = nn.Conv2d(in_channels[-1], out_channels, 1)
        self.conv1by1_4 = nn.Conv2d(in_channels[-2], out_channels, 1)
        self.conv1by1_3 = nn.Conv2d(in_channels[-3], out_channels, 1)
        self.conv1by1_2 = nn.Conv2d(in_channels[-4], out_channels, 1)
        self.conv1by1_1 = nn.Conv2d(in_channels[-5], out_channels, 1)


    def forward(self, x):
        f5 = x[-1]  #####[16,512,32,32]  (no upsampling)
        f4 = x[-2]  #####[16,256,32,32]  (no upsampling)
        f3 = x[-3]  #####[16,128,32,32]
        f2 = x[-4]  #####[16,64,64,64]
        f1 = x[-5]  #####[16,64,64,64]
    

        # f4 = self.conv1by1_4(f4) 
        
        f4 = f5 + self.conv1by1_4(f4) #####[16,512,32,32] 
        f4 = F.interpolate(f4, scale_factor= 2 , mode="nearest")  #####[16,512,64,64]  
        
        f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")  #####[16,128,64,64]
        f3 = self.conv1by1_3(f3) + f4                             #####[16,512,64,64]  
        # f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")

        f2 = self.conv1by1_2(f2) + f3
        # f2 = F.interpolate(f2, scale_factor= 2 , mode="nearest")

        f1 = self.conv1by1_1(f1) + f2
        # print("f1hou:", f1.size())

        return f1

class FPN_S3(nn.Module):
    """resnet(S3) + FPN

    return: 融合后的特征层
    """
    def __init__(self, in_channels = [64, 64, 128], out_channels = 256):
        super().__init__()

        self.conv1by1_3 = nn.Conv2d(in_channels[-1], out_channels, 1)
        self.conv1by1_2 = nn.Conv2d(in_channels[-2], out_channels, 1)
        self.conv1by1_1 = nn.Conv2d(in_channels[-3], out_channels, 1)


    def forward(self, x):
        f3 = x[-1]  #####[16,128,32,32]
        f2 = x[-2]  #####[16,64,64,64]
        f1 = x[-3]  #####[16,64,64,64]
    
        
        f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")  #####[16,128,64,64]
        f3 = self.conv1by1_3(f3)                            #####[16,256,64,64]  

        f2 = self.conv1by1_2(f2) + f3
        # f2 = F.interpolate(f2, scale_factor= 2 , mode="nearest")

        f1 = self.conv1by1_1(f1) + f2
        # print("f1hou:", f1.size())

        return f1        

class FPN_S2(nn.Module):
    """resnet(S2) + FPN

    return: 融合后的特征层
    """
    def __init__(self, in_channels = [64, 64], out_channels = 256):
        super().__init__()

        self.conv1by1_2 = nn.Conv2d(in_channels[-1], out_channels, 1)
        self.conv1by1_1 = nn.Conv2d(in_channels[-2], out_channels, 1)


    def forward(self, x):
        f2 = x[-1]  #####[16,64,64,64]
        f1 = x[-2]  #####[16,64,64,64]

        f1 = self.conv1by1_1(f1) + self.conv1by1_2(f2)
    
        return f1        

class Diff(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.con1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.con2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.con1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.con2(x)
        x = self.act(x)

        return x