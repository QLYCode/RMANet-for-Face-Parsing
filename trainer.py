import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from rmanet import rmanet
from utils import *
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/training')


class InfluenceBalancedLoss(nn.Module):
    def __init__(self, num_classes=19, ignore_index=-1, reduction='mean', alpha=0.5, beta=1.0):

        super(InfluenceBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):

        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # [batch_size*height*width, num_classes]
        targets = targets.view(-1)  # [batch_size*height*width]

        # Mask out the ignore_index
        valid_mask = (targets != self.ignore_index)
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        # Compute class frequencies (influence)
        class_counts = torch.bincount(targets, minlength=self.num_classes).float()
        class_influence = class_counts / class_counts.sum()  # Normalize by total count

        # Apply smoothing factor to avoid division by zero
        class_influence = torch.clamp(class_influence, min=self.beta)

        # Cross-entropy loss with re-weighting based on influence
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute the influence-balanced weight
        weights = self.alpha / (class_influence[targets] + self.beta)

        # Apply the weights to the cross-entropy loss
        weighted_loss = weights * ce_loss
        print(weighted_loss.shape)
        # Return the final loss, reduced by mean or sum
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel
        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        self.use_tensorboard = config.use_tensorboard
        self.img_path = config.img_path
        self.label_path = config.label_path 
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            self.G.train()
            try:
                imgs, labels = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                imgs, labels = next(data_iter)

            size = labels.size()
            labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
            labels_real_plain = labels[:, 0, :, :].cuda()
            labels = labels[:, 0, :, :].view(size[0], 1, size[2], size[3])
            oneHot_size = (size[0], 19, size[2], size[3])
            labels_real = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            labels_real = labels_real.scatter_(1, labels.data.long().cuda(), 1.0)
            imgs = imgs.cuda()

            # ================== Train G =================== #
            labels_predict = self.G(imgs).cuda()

            # The first training phase: cross entropy loss
            c_loss = cross_entropy2d(labels_predict, labels_real_plain.long()).cuda()

            # The second training phase:  IB loss
            IB_loss = InfluenceBalancedLoss(num_classes=19, ignore_index=-1, reduction='mean', alpha=0.5, beta=1.0).cuda()
            loss = IB_loss(labels_predict, labels_real_plain.long()).cuda()

            self.reset_grad()
            loss.backward()
            self.g_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], total_loss: {:.4f}".
                      format(elapsed, step + 1, self.total_step, loss.data))

            # scalr info on tensorboardX
            writer.add_scalar('Loss', loss.data, step)

            # image infor on tensorboardX
            #img_combine = imgs[0]
            #real_combine = label_batch_real[0]
            #predict_combine = label_batch_predict[0]
            #for i in range(1, self.batch_size):
                #img_combine = torch.cat([img_combine, imgs[i]], 2)
                #real_combine = torch.cat([real_combine, label_batch_real[i]], 2)
                #predict_combine = torch.cat([predict_combine, label_batch_predict[i]], 2)
            # writer.add_image('imresult/img', (img_combine.data + 1) / 2.0, step)
            # writer.add_image('imresult/real', real_combine, step)
            # writer.add_image('imresult/predict', predict_combine, step)

            # Sample images
            if (step + 1) % 5000 == 0:
                labels_sample = self.G(imgs)
                labels_sample = generate_label(labels_sample,self.imsize)
                labels_sample = labels_sample
                save_image(denorm(labels_sample.data),
                           os.path.join(self.sample_path, '{}_predict.png'.format(step + 1)))

            if (step+1) % 1000==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
    
    def build_model(self):

        self.G = rmanet().cuda()

        """
        # The second training phase
        rmanet_weights = torch.load('1000_G.pth')
        mapped_weights = {}
        for k_model, k_segnet in zip(rmanet_weights.keys(), self.G.state_dict().keys()):
            if "features" in k_model:
                mapped_weights[k_segnet] = rmanet_weights[k_model]
        try:
            print('loaded trained models (step: {})..!'.format("32000_G.pth"))
            self.G.load_state_dict(mapped_weights)
        except:
            pass

        """

        if self.parallel:
            self.G = nn.DataParallel(self.G)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])

        # print networks
        print(self.G)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))


    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))


