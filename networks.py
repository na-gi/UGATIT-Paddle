import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear, InstanceNorm, PRelu, SpectralNorm
from paddle.fluid.dygraph import Sequential


class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        # 先通过一个卷积核尺寸为7的卷积层，图片大小不变，通道数变为64
        DownBlock += [ReflectionPad2d(3),
                      Conv2D(input_nc, ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(ngf),
                      PRelu(mode="all")]

        # Down-Sampling --> 下采样模块
        n_downsampling = 2
        # 两层下采样，img_size缩小4倍（64），通道数扩大4倍（256）
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [ReflectionPad2d(1),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(ngf * mult * 2),
                          PRelu(mode="all")]

        # Down-Sampling Bottleneck  --> 编码器中的残差模块
        mult = 2**n_downsampling
        # 6个残差块，尺寸和通道数都不变
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map --> 产生类别激活图
        # 接着global average pooling后的全连接层
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        # 接着global max pooling后的全连接层
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        #下面1x1卷积和激活函数，是为了得到两个pooling合并后的特征图
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True, act='relu')
        # self.relu = nn.ReLU(True)

        # Gamma, Beta block --> 生成自适应 L-B Normalization(AdaILN)中的Gamma, Beta
        # 确定轻量级，FC使用的是两个256 --> 256的全连接层
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                #   nn.ReLU(True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                #   nn.ReLU(True)
            ]
        else:
            # 不是轻量级，则下面的1024x1024 --> 256的全连接层和一个256 --> 256的全连接层
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                #   nn.ReLU(True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                #   nn.ReLU(True)
            ]
        # AdaILN中的Gamma, Beta
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck --> 解码器中的自适应残差模块
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling --> 解码器中的上采样模块
        UpBlock2 = []
        # 上采样与编码器的下采样对应
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [Upsample(),
                         ReflectionPad2d(1),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False, act='relu'),
                         ILN(int(ngf * mult / 2)),  # 注:只有自适应残差块使用AdaILN
                        #  nn.ReLU(True)
            ]
        # 最后一层卷积层，与最开始的卷积层对应
        UpBlock2 += [ReflectionPad2d(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False, act='tanh'),
                    #  nn.Tanh()
        ]

        self.DownBlock = Sequential(*DownBlock) # 编码器整个模块
        self.FC = Sequential(*FC)               # 生成gamma,beta的全连接层模块
        self.UpBlock2 = Sequential(*UpBlock2)   # 只包含上采样后的模块，不包含残差块

    def forward(self, input):
        x = self.DownBlock(input)
        # 得到编码器的输出,对应途中encoder feature map
        # torch.Size([1, 256, 64, 64])
        # gap torch.Size([1, 256, 1, 1])

        gap = Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='avg')(x)  #全局平均池化
        gap = fluid.layers.reshape(gap, shape=[x.shape[0], -1]) #torch.Size([1, 1])
        gap_logit = self.gap_fc(gap)     #gap的预测
        gap_weight = list(self.gap_fc.parameters())[0] #self.gap_fc的权重参数 torch.Size([1, 256])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[0])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[3])
        gap = x * gap_weight    #得到全局平均池化加持权重的特征图 torch.Size([1, 256, 64, 64])

        gmp = Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='max')(x)
        gmp = fluid.layers.reshape(gmp, shape=[x.shape[0], -1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[0])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[3])     
        gmp = x * gmp_weight  #torch.Size([1, 256, 64, 64])

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)   #结合gap和gmp的cam_logit预测
        x = fluid.layers.concat([gap, gmp], 1)   #torch.Size([1, 512, 64, 64])      
        x = self.conv1x1(x) #接入一个卷积层，通道数512转换为256 torch.Size([1, 256, 64, 64])
        #x = self.relu(self.conv1x1(x))
        #torch.Size([1, 256, 64, 64])

        # heatmap = torch.sum(x, dim=1, keepdim=True)
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  #得到注意力热力图
        #heatmap torch.Size([1, 1, 64, 64])

        if self.light:
            #轻量级则先经过一个gap
            x_ = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
            x_ = fluid.layers.reshape(x_, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        else:
            x_=fluid.layers.reshape(x, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        gamma, beta = self.gamma(x_), self.beta(x_) #得到自适应gamma和beta
        # gamma torch.Size([1, 256]) beta torch.Size([1, 256])

        for i in range(self.n_blocks):
            # 将自适应gamma和beta送入到AdaILN
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)  #通过上采样后的模块，得到生成结果
        #out torch.Size([1, 3, 256, 256]) cam_logit torch.Size([1, 2])  heatmap torch.Size([1, 1, 64, 64])

        return out, cam_logit, heatmap  #模型输出为生成结果，cam预测以及热力图


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim),
                       PRelu(mode="all")]

        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim)]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        # self.relu1 = nn.ReLU(True)

        self.pad2 = ReflectionPad2d(1)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = fluid.layers.relu(out)
        # out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x

# Adaptive Layer-Instance Normalization代码
class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        # adaILN的参数p，通过这个参数来动态调整LN和IN的占比
        self.rho = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.9, dtype='float32')

    def forward(self, input, gamma, beta):
        # torch.Size([1, 256, 64, 64])
        ninput = input.numpy()
        # 先求两种规范化的值
        in_mean, in_var = np.mean(ninput, axis=(2, 3), keepdims=True), np.var(ninput, axis=(2, 3), keepdims=True)
        out_in = (ninput - in_mean) / np.sqrt(in_var + self.eps)
        ln_mean, ln_var = np.mean(ninput, axis=(1, 2, 3), keepdims=True), np.var(ninput, axis=(1, 2, 3), keepdims=True)
        out_ln = (ninput - ln_mean) / np.sqrt(ln_var + self.eps)
        out_in = fluid.dygraph.base.to_variable(out_in)
        out_ln = fluid.dygraph.base.to_variable(out_ln)
        ninput = fluid.dygraph.base.to_variable(ninput)
        #out = fluid.dygraph.base.to_variable(out)
        # 合并两种规范化(IN, LN)
        out = self.rho * out_in + (1-self.rho) * out_ln
        # 扩张得到结果
        gamma = fluid.layers.unsqueeze(input=gamma, axes=[2])
        gamma = fluid.layers.unsqueeze(input=gamma, axes=[3])   
        beta = fluid.layers.unsqueeze(input=beta, axes=[2])
        beta = fluid.layers.unsqueeze(input=beta, axes=[3])  
        out = out * gamma + beta
        # out torch.Size([1, 256, 64, 64])

        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.0, dtype='float32')
        self.gamma = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=1.0, dtype='float32')
        self.beta = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.0, dtype='float32')

    def forward(self, input):
        #torch.Size([1, 128, 128, 128])
        ninput = input.numpy()

        in_mean, in_var=np.mean(ninput, axis=(2, 3), keepdims=True), np.var(ninput, axis=(2, 3), keepdims=True)
        out_in = (ninput - in_mean) / np.sqrt(in_var + self.eps)
        ln_mean, ln_var = np.mean(ninput, axis=(1, 2, 3), keepdims=True), np.var(ninput, axis=(1, 2, 3), keepdims=True)
        out_ln = (ninput - ln_mean) / np.sqrt(ln_var + self.eps)
        out_in = fluid.dygraph.base.to_variable(out_in)
        out_ln = fluid.dygraph.base.to_variable(out_ln)
        ninput = fluid.dygraph.base.to_variable(ninput)        
        out = self.rho * out_in + (1-self.rho) * out_ln
        out = out * self.gamma + self.beta
        # out torch.Size([1, 128, 128, 128])

        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        # 第一层下采样, 尺寸减半(128)，通道数为64
        model = [ReflectionPad2d(1),
                 Spectralnorm(Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=1, bias_attr=True, act='leaky_relu')),
                #  nn.LeakyReLU(0.2, True)
        ]
        # 第二，三层下采样，尺寸再缩4倍(32)，通道数为256
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d(1),
                      Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=True, act='leaky_relu')),
                    #   nn.LeakyReLU(0.2, True)
            ]
        # 尺寸不变（32），通道数为512
        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d(1),
                  Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=True, act='leaky_relu')),
                #   nn.LeakyReLU(0.2, True)
        ]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)
        # self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = ReflectionPad2d(1)
        self.conv = Spectralnorm(Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False))

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input)   #[1, 2048, 2, 2]

        gap = Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='avg')(x) #[1, 2048, 1, 1]
        gap = fluid.layers.reshape(gap, shape=[x.shape[0], -1]) 
        gap_logit = self.gap_fc(gap)    #torch.Size([1, 1])
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[0])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[3])   
        gap = x * gap_weight #[1, 2048, 2, 2]

        gmp = Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='max')(x)
        gmp = fluid.layers.reshape(gmp, shape=[x.shape[0], -1])        
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[0])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[3])          
        gmp = x * gmp_weight

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = fluid.layers.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w


class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size
    
    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode='reflect')


# 定义上采样模块
class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, actual_shape=out_shape)
        return out


class Spectralnorm(fluid.dygraph.Layer):
    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        
        return out

    
class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out