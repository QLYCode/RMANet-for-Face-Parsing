from model_utils import *


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, dropout_rate=0.0):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU())
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.InstanceNorm2d(output_dim),
        )

    def forward(self, x):
        return  (self.conv_block(x) + self.conv_skip(x))


class ASPP(nn.Module):
    def __init__(self, in_channel=1024, depth=256, stride=1):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveMaxPool2d(32)
        self.atrous_block1 = unetConv2(in_channel, depth, kernel_size =3, is_batchnorm=True)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, stride, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, stride, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, stride, padding=18, dilation=18)
        self.fusion = nn.Conv2d(depth, depth, 1, stride)

    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        adaptivemaxpool = self.mean(atrous_block1)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = atrous_block1 + self.fusion(atrous_block6 + atrous_block12+atrous_block18 + adaptivemaxpool)
        return net

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, outsize, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(outsize)
        self.max_pool = nn.AdaptiveMaxPool2d(outsize)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class rmanet(nn.Module):
    def __init__(
            self,
            feature_scale=4,
            n_classes=19,
            is_deconv=True,
            in_channels=3,
            is_batchnorm=True,
    ):
        super(rmanet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        filters = [64, 128, 256, 384, 768, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = nn.Sequential(ResidualConv(self.in_channels, filters[0], stride=1, padding=1,dropout_rate=0.0),
                                   ResidualConv(filters[0], filters[0], stride=1, padding=1,dropout_rate=0.0))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(ResidualConv(filters[0], filters[1], stride=1, padding=1,dropout_rate=0.1),
                                   ResidualConv(filters[1], filters[1], stride=1, padding=1,dropout_rate=0.1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(ResidualConv(filters[1], filters[2], stride=1, padding=1,dropout_rate=0.1),
                                   ResidualConv(filters[2], filters[2], stride=1, padding=1,dropout_rate=0.1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Sequential(ResidualConv(filters[2], filters[3], stride=1, padding=1,dropout_rate=0.1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.ca0 = ChannelAttention(filters[0], outsize=512, ratio=16)
        self.ca1 = ChannelAttention(filters[1], outsize=256, ratio=16)
        self.ca2 = ChannelAttention(filters[2], outsize=128, ratio=16)
        self.ca3 = ChannelAttention(filters[3], outsize=64, ratio=16)

        self.aspp = ASPP(in_channel=filters[3], depth=filters[4], stride=1)


        self.conv_1 = nn.Sequential(nn.Conv2d(64, 48, 3, 1, 1),
                                   nn.InstanceNorm2d(48),
                                   nn.ReLU())

        self.conv_2 = nn.Sequential(nn.Conv2d(48, 64, 3, 1, 1),
                                  nn.InstanceNorm2d(64),
                                  nn.ReLU())

        self.conv_3 = unetConv2(32, 16, 3, is_batchnorm)
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], 48, self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0)

        self.final = nn.Conv2d(32, n_classes, 1)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        ca1 = self.ca0(conv1)
        ca2 = self.ca1(conv2)
        ca3 = self.ca2(conv3)
        ca4 = self.ca3(conv4)

        center = self.aspp(maxpool4)

        up4 = self.up_concat4(ca4, center)
        up3 = self.conv_2(self.up_concat3(self.conv_1(ca3), up4))
        up2 = self.up_concat2(ca2, up3)
        up1 = torch.cat([ca1,self.conv_3(self.up_concat1(up2))], 1)
        final = self.final(up1)
        return final


class Discriminator(nn.Module):
    def __init__(self, in_channels=4, out_channels=19):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, out_channels, kernel_size=7)
        )

    def forward(self, x, x1):
        x = torch.cat((x, x1), dim=1)
        return self.model(x)




if __name__ == '__main__':
    model =  rmanet()
    number = sum(p.numel() for p in model.parameters())
    print(number)


