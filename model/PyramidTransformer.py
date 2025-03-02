import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample
import numpy as np
import random

torch.set_num_threads(50)

class Dense(nn.Module):
    def __init__(self, k=7, layers=4, fm=32, outfm=None, f=nn.ReLU(inplace=True)):
        super(Dense, self).__init__()
        p = (k-1)/2
        self.f = f
        self.mlist = nn.ModuleList([nn.Conv2d(fm * (i+1), fm if outfm is None or i < layers - 1 else outfm, k, padding=p) for i in range(layers)])

    def forward(self, x):
        outputs = [x]
        for m in self.mlist:
            if m != self.mlist[-1]:
                outputs.append(self.f(m(torch.cat(outputs, 1))))
            else:
                outputs.append(m(torch.cat(outputs, 1)))
        return outputs[-1]

class PG(nn.Module):
    def __init__(self, k=7, f=nn.ReLU(inplace=True)):
        super(PG, self).__init__()
        print('building PG with kernel', k)
        p = (k-1)/2
        self.f = f
        self.encode = nn.Conv2d(2, 64, k, padding=p, groups=2)
        self.dense1 = Dense(fm=64, outfm=32)
        self.dense2 = Dense(fm=32, outfm=16, k=5)
        self.dense3 = Dense(fm=16, k=3)
        self.decode = nn.Conv2d(16, 2, 3, padding=1)

    def forward(self, x):
        embedding = self.f(self.encode(x))
        out1 = self.dense1(embedding)
        out2 = self.dense2(out1)
        out3 = self.dense3(out2)
        out = self.decode(out3).permute(0,2,3,1)
        return out / 10

class G(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)

    def __init__(self, k=7, f=nn.LeakyReLU(inplace=True), infm=2):
        super(G, self).__init__()
        print('building G with kernel', k)
        p = (k-1)//2
        self.conv1 = nn.Conv2d(infm, 32, k, padding=p)
        self.conv2 = nn.Conv2d(32, 64, k, padding=p)
        self.conv3 = nn.Conv2d(64, 32, k, padding=p)
        self.conv4 = nn.Conv2d(32, 16, k, padding=p)
        self.conv5 = nn.Conv2d(16, 2, k, padding=p)
        self.seq = nn.Sequential(self.conv1, f,
                                 self.conv2, f,
                                 self.conv3, f,
                                 self.conv4, f,
                                 self.conv5)
        self.initc(self.conv1)
        self.initc(self.conv2)
        self.initc(self.conv3)
        self.initc(self.conv4)
        self.initc(self.conv5)

    def forward(self, x):
        return self.seq(x).permute(0,2,3,1) / 10

class MPG(nn.Module):
    def __init__(self, k=7, f=nn.ReLU()):
        super(MPG, self).__init__()
        print('building MPG with kernel', k)
        p = (k-1)/2
        self.mp = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv1 = nn.Conv2d(2, 32, 7, padding=3, groups=2)
        self.conv2 = nn.Conv2d(32, 64, 7, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 5, padding=2)
        self.conv5 = nn.Conv2d(16, 2, 3, padding=1)
        self.seq = nn.Sequential(self.conv1, f,
                                 self.conv2, f, self.mp,
                                 self.conv3, f, self.mp,
                                 self.conv4, f,
                                 self.conv5, self.up)

    def forward(self, x, vis=None):
        return self.seq(x).permute(0,2,3,1)

class DeepG(nn.Module):
    def initc(self, m):
        m.weight.data *= np.sqrt(6)

    def __init__(self, k=7, infm=2, f=nn.ReLU()):
        super(DeepG, self).__init__()
        print('building G with kernel', k)
        p = (k-1)/2
        self.conv1 = nn.Conv2d(infm, 32, 7, padding=3, groups=2)
        self.conv2a = nn.Conv2d(32, 32, 7, padding=3)
        self.conv2b = nn.Conv2d(32, 32, 7, padding=3)
        self.conv3a = nn.Conv2d(32, 64, 7, padding=3)
        self.conv3b = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4a = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4b = nn.Conv2d(64, 32, 5, padding=2)
        self.conv5a = nn.Conv2d(32, 16, 3, padding=1)
        self.conv5b = nn.Conv2d(16, 2, 3, padding=1)
        self.seq = nn.Sequential(self.conv1, f,
                                 self.conv2a, f,
                                 self.conv2b, f,
                                 self.conv3a, f,
                                 self.conv3b, f,
                                 self.conv4a, f,
                                 self.conv4b, f,
                                 self.conv5a, f,
                                 self.conv5b)
        for m in self.seq.children():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                self.initc(m)

    def forward(self, x, vis=None):
        return self.seq(x).permute(0,2,3,1)

class DConv(nn.Module):
    def __init__(self, infm, outfm, k, padding, dilation=1, groups=1, f=nn.ReLU(inplace=True)):
        assert infm == outfm

        super(DConv, self).__init__()
        self.f = f
        self.conv = nn.Conv2d(infm, outfm, k, padding=padding, groups=groups, dilation=dilation)
        weights = torch.zeros((outfm, infm, k, k)).normal_(0, 0.01)
        for i in range(infm):
            weights[i,i,k//2,k//2] = 1
        self.conv.weight = nn.Parameter(weights)
        self.conv.bias.data /= 10

    def forward(self, x):
        return self.f(self.conv(x))

if __name__ == '__main__':
    x = Variable(torch.zeros((1,4,100,100)))
    conv = DConv(4, 4, 3, 1)
    print(conv(x).size())

class DG(nn.Module):
    def __init__(self, k=3, f=nn.ReLU(), t=1):
        super(DG, self).__init__()
        print('building DG with kernel', k, ' targets:', t)
        p = (k-1)//2
        d = (k+1)//2
        self.f = f
        fm = 32 * (t+1)
        self.conv1 = nn.Conv2d(t+1, fm, k, padding=p, groups=t+1)
        self.conv2 = nn.Conv2d(fm, fm, k, padding=p)
        self.conv3 = DConv(fm, fm, k, padding=p*d, dilation=d)
        self.conv4 = DConv(fm, fm, k, padding=p*d*2, dilation=d*2)
        self.conv5 = DConv(fm, fm, k, padding=p*d*4, dilation=d*4)
        self.conv6 = DConv(fm, fm, k, padding=p*d*8, dilation=d*8)
        self.conv7 = nn.Conv2d(fm, 16, 3, padding=1)
        self.conv8 = nn.Conv2d(16, 2, 3, padding=1)
        self.conv8.weight.data /= 10
        self.conv8.bias.data /= 10

    def forward(self, x, vis=False):
        out = self.f(self.conv1(x))
        if vis:
            idd = 'vis' + str(random.randint(0,100)) + '_'
            gif(idd + '1', np.squeeze(out.data.cpu().numpy()) * 255)
        out = self.f(self.conv2(out))
        if vis:
            idd = 'vis' + str(random.randint(0,100)) + '_'
            gif(idd + '2', np.squeeze(out.data.cpu().numpy()) * 255)
        out = self.f(self.conv3(out))
        if vis:
            idd = 'vis' + str(random.randint(0,100)) + '_'
            gif(idd + '3', np.squeeze(out.data.cpu().numpy()) * 255)
        out = self.f(self.conv4(out))
        if vis:
            idd = 'vis' + str(random.randint(0,100)) + '_'
            gif(idd + '4', np.squeeze(out.data.cpu().numpy()) * 255)
        out = self.f(self.conv5(out))
        if vis:
            idd = 'vis' + str(random.randint(0,100)) + '_'
            gif(idd + '5', np.squeeze(out.data.cpu().numpy()) * 255)
        out = self.f(self.conv6(out))
        if vis:
            idd = 'vis' + str(random.randint(0,100)) + '_'
            gif(idd + '6', np.squeeze(out.data.cpu().numpy()) * 255)
        out = self.f(self.conv7(out))
        if vis:
            idd = 'vis' + str(random.randint(0,100)) + '_'
            gif(idd + '7', np.squeeze(out.data.cpu().numpy()) * 255)
        out = self.f(self.conv8(out))
        if vis:
            idd = 'vis' + str(random.randint(0,100)) + '_'
            gif(idd + '8', np.squeeze(out.data.cpu().numpy()) * 255)
        return out.permute(0,2,3,1)

class Pyramid(nn.Module):
    def get_identity_grid(self, dim):
        if dim not in self.identities:
            gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
            I = np.stack(np.meshgrid(gx, gy))
            I = np.expand_dims(I, 0)
            I = torch.FloatTensor(I)
            I = torch.autograd.Variable(I, requires_grad=False)
            I = I.permute(0,2,3,1)
            if self.gpu:
                self.identities[dim] = I.cuda()
            else:
                self.identities[dim] = I

        return self.identities[dim]

    def __init__(self, size, dim, skip, k, dilate=False, amp=False, unet=False, num_targets=1, gpu=False):
        super(Pyramid, self).__init__()
        rdim = dim // (2 ** (size))
        print('------- Constructing PyramidNet with size', size, '(' + str(size-1) + ' downsamples)')
        self.identities = {}
        self.gpu  = gpu
        self.skip = skip
        self.size = size

        if dilate:
            if amp:
                self.mlist = nn.ModuleList([AmpDG(k=k) for level in range(size)])
            else:
                self.mlist = nn.ModuleList([DG(k=k, t=num_targets) for level in range(size)])
        elif unet:
            self.mlist = nn.ModuleList([UNet(k=3, depth=3) for level in range(size)])
        else:
            self.mlist = nn.ModuleList([DeepG(k=k) for level in range(size)])
        self.f_up = lambda x: nn.Upsample(scale_factor=x, mode='bilinear')
        self.up = self.f_up(2)
        self.down = nn.AvgPool2d(2, 2)
        self.I = self.get_identity_grid(rdim)
        self.Zero = self.I - self.I

    def forward(self, stack, idx=0, vis=None):
        if idx < self.size:
            field_so_far, residuals_so_far = self.forward(self.down(stack), idx + 1, vis) # (B,dim,dim,2)
            field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)
        else:
            return self.I, [ self.I ]

        if idx < self.skip:
            residuals_so_far.insert(0, self.f_up(2 ** (self.size - idx))(self.Zero.permute(0,3,1,2)).permute(0,2,3,1)) # placeholder
            return field_so_far, residuals_so_far
        else:
            resampled_source = grid_sample(stack[:,0:1], field_so_far)
            new_input_stack = torch.cat((resampled_source, stack[:,1:]),1)
            if vis is not None:
                idd = 'stack' + str(hash(self)) + '_' + str(idx)
                gif(idd, np.squeeze(new_input_stack.data.cpu().numpy()) * 255)
            residual = self.mlist[idx](new_input_stack, vis==idx)
            residuals_so_far.insert(0, residual)
            return residual + field_so_far, residuals_so_far

class Encoder(nn.Module):
    def __init__(self, fm=8, size=5):
        super(Encoder, self).__init__()
        self.enclist = nn.ModuleList([Enc(i==0, fm=fm) for i in range(size)])
        self.declist = nn.ModuleList([Dec(i==size-1, fm=fm) for i in range(size)])
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        out = x
        for idx, enc in enumerate(self.enclist):
            out = self.down(enc(out)[0])
        for dec in self.declist:
            out = self.up(dec(out))
        return out

class Dec(nn.Module):
    def initw(self, m):
        k = m.weight.size(2)
        infm, outfm = m.weight.size(1), m.weight.size(0)
        weights = torch.zeros(m.weight.size()).normal_(0, 0.1)
        for outfm_i in range(m.weight.size(0)):
            for infm_i in range(m.weight.size(1)):
                weights[outfm_i,infm_i,k//2,k//2] = 1.0 / infm
        m.weight = nn.Parameter(weights)
        m.bias.data = torch.abs(m.bias.data) / 100.0

    def __init__(self, bottom, fm=8):
        super(Dec, self).__init__()
        self.bottom = bottom
        self.f = nn.ReLU()
        self.c1 = nn.Conv2d(fm, fm, 3, padding=1)
        self.c2 = nn.Conv2d(fm, fm if not bottom else 1, 3, padding=1)
        self.initw(self.c1)
        self.initw(self.c2)

    def forward(self, x):
        out = torch.cat((self.f(self.c1(x[:,0:x.size(1)//2])), self.f(self.c1(x[:,x.size(1)//2:]))), 1)
        f = (lambda x: x) if self.bottom else self.f
        out = torch.cat((f(self.c2(out[:,0:out.size(1)//2])), f(self.c2(out[:,out.size(1)//2:]))), 1)
        return out

class Enc(nn.Module):
    def initw(self, m):
        k = m.weight.size(2)
        infm, outfm = m.weight.size(1), m.weight.size(0)
        weights = torch.zeros(m.weight.size()).normal_(0, 0.1)
        for outfm_i in range(m.weight.size(0)):
            for infm_i in range(m.weight.size(1)):
                weights[outfm_i,infm_i,k//2,k//2] = 1.0 / infm
        m.weight = nn.Parameter(weights)
        m.bias.data = torch.abs(m.bias.data) / 100.0

    def initr(self, m):
        weights = torch.zeros(m.weight.size()).normal_(0, 0.04)
        weights[0,2,0,0] = 1
        m.weight.data = weights
        m.bias.data = torch.abs(m.bias.data) / 10.0

    def initc(self, m):
        m.weight.data *= np.sqrt(6)

    def __init__(self, bottom, infm=8, outfm=None):
        super(Enc, self).__init__()
        if not outfm:
            outfm = infm
        self.f = nn.LeakyReLU(inplace=True)
        self.c1 = nn.Conv2d(1 if bottom else infm, infm, 3, padding=1)
        self.c2 = nn.Conv2d(infm, outfm, 3, padding=1)
        self.initc(self.c1)
        self.initc(self.c2)

    def forward(self, x):
        out1 = torch.cat((self.f(self.c1(x[:,0:x.size(1)//2])), self.f(self.c1(x[:,x.size(1)//2:]))), 1)
        out2 = torch.cat((self.f(self.c2(out1[:,0:out1.size(1)//2])), self.f(self.c2(out1[:,out1.size(1)//2:]))), 1)
        return out2

class EPyramid(nn.Module):
    def get_identity_grid(self, dim):
        if dim not in self.identities:
            gx, gy = np.linspace(-1, 1, dim), np.linspace(-1, 1, dim)
            I = np.stack(np.meshgrid(gx, gy))
            I = np.expand_dims(I, 0)
            I = torch.FloatTensor(I)
            I = torch.autograd.Variable(I, requires_grad=False)
            I = I.permute(0,2,3,1)
            if self.gpu:
                self.identities[dim] = I.cuda()
            else:
                self.identities[dim] = I

        return self.identities[dim]

    def __init__(self, size, dim, skip, k, dilate=False, amp=False, unet=False, num_targets=1, name=None, gpu=False):
        super(EPyramid, self).__init__()
        dim=1280
        rdim = dim // (2 ** (size - 1))
        print('------- Constructing EPyramidNet with size', size, '(' + str(size-1) + ' downsamples) ' + str(dim))
        self.name = name
        fm = 6
        self.identities = {}
        self.gpu  = gpu
        self.skip = skip
        self.size = size
        self.mlist = nn.ModuleList([G(k=k, infm=fm*(level+2)*2) for level in range(size)])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.down = nn.MaxPool2d(2)
        self.enclist = nn.ModuleList([Enc(level==0, infm=fm*(level+1), outfm=fm*(level+2)) for level in range(size)])
        self.I = self.get_identity_grid(rdim)
        self.Zero = self.I - self.I
        self.counter = 0

    def forward(self, stack, target_level, vis=False):
        encodings = [self.enclist[0](stack)]
        for idx in range(1, self.size):
            encodings.append(self.enclist[idx](self.down(encodings[-1])))

        residuals = [self.I]
        field_so_far = self.I
        for i in range(self.size - 1, target_level - 1, -1):
            inputs_i = encodings[i]
            resampled_source = grid_sample(inputs_i[:,0:inputs_i.size(1)//2], field_so_far)
            new_input_i = torch.cat((resampled_source, inputs_i[:,inputs_i.size(1)//2:]), 1)
            rfield = self.mlist[i](new_input_i)
            residuals.append(rfield)
            field_so_far = rfield + field_so_far
            if i != target_level:
                field_so_far = self.up(field_so_far.permute(0,3,1,2)).permute(0,2,3,1)

        return field_so_far, residuals

class PyramidTransformer(nn.Module):
    def __init__(self, size=4, dim=192, skip=0, k=7, dilate=False, amp=False, unet=False, num_targets=1, name=None, gpu=False):
        super(PyramidTransformer, self).__init__()
        self.pyramid = EPyramid(size, dim, skip, k, dilate, amp, unet, num_targets, name=name, gpu=gpu)

    def open_layer(self):
        if self.pyramid.skip > 0:
            self.pyramid.skip -= 1
            print('Pyramid now using', self.pyramid.size - self.pyramid.skip, 'layers.')

    def select_module(self, idx):
        for g in self.pyramid.mlist:
            g.requires_grad = False
        self.pyramid.mlist[idx].requires_grad = True

    def select_all(self):
        for g in self.pyramid.mlist:
            g.requires_grad = True

    def forward(self, x, idx=0, vis=None):
        field, residuals = self.pyramid(x, idx, vis)
        #factor = 8
        #I =  self.pyramid.get_identity_grid(field.size()[2])
        #irfield = field - I
        #irfield_smooth = nn.AvgPool2d(2**factor+1, stride=1, padding=2**(factor-1), count_include_pad=False)(irfield.permute(0,3,1,2)).permute(0,2,3,1)
        #field = irfield_smooth + I
        return grid_sample(x[:,0:1,:,:], field), field, residuals

    ################################################################
    # Begin Sergiy API
    ################################################################

    @staticmethod
    def load(archive_path=None, height=5, dim=1024, skips=0, k=7, cuda=True, dilate=False, amp=False, unet=False, num_targets=1, name=None):
        """
        Builds and load a model with the specified architecture from
        an archive.

        Params:
            height: the number of layers in the pyramid (including
                    bottom layer (number of downsamples = height - 1)
            dim:    the size of the full resolution images used as input
            skips:  the number of residual fields (from the bottom of the
                    pyramid) to skip
            cuda:   whether or not to move the model to the GPU
        """
        assert archive_path is not None, "Must provide an archive"

        model = PyramidTransformer(size=height, dim=dim, k=k, skip=skips, dilate=dilate, amp=amp, unet=unet, num_targets=num_targets, name=name, gpu=cuda)
        if cuda:
            model = model.cuda()
        for p in model.parameters():
            p.requires_grad = False
        model.train(False)

        print('Loading model state from', archive_path + '...')
        model.load_state_dict(torch.load(archive_path, map_location=lambda storage, loc: storage))

        return model

    def apply(self, source, target, skip=0, vis=None):
        """
        Applies the model to an input. Inputs (source and target) are
        expected to be of shape (dim // (2 ** skip), dim // (2 ** skip)),
        where dim is the argument that was passed to the constructor.

        Params:
            source: the source image (to be transformed)
            target: the target image (image to align the source to)
            skip:   resolution at which alignment should occur.
        """
        source = source.unsqueeze(0)
        if len(target.size()) == 2:
            target = target.unsqueeze(0)
        return self.forward(torch.cat((source,target), 0).unsqueeze(0), idx=skip, vis=vis)
