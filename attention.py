import torch
import torch.nn as nn


class ExampleNet(nn.Module):
    def __init__(self, num_classes):
        super(ExampleNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.AdaptiveMaxPool2d((1, 1)))
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x):
        feats = self.features(x)
        cls = self.classifier(feats.squeeze(-1).squeeze(-1))
        return cls


# https://openreview.net/pdf?id=rJe7FW-Cb
class Attention(nn.Module):
    # def __init__(self, N, K, height, width, num_classes):
    def __init__(self, K, num_classes):
        super(Attention, self).__init__()
        self.N = None
        self.K = K
        self.softmax = nn.Softmax2d()
        self.num_classes = num_classes
        self.tanh = nn.Tanh()
    def forward(self, x):
        ############################
        # figure out size of filters from input
        ############################
        if self.N is None:
            self.N = x.data.numpy().shape[1]
            self.height = x.data.numpy().shape[2]
            self.width = x.data.numpy().shape[3]
            self.conv = nn.Conv2d(self.N, self.K, kernel_size=1, bias=False)
            self.dimred_conv = nn.Conv2d(self.N*self.K, self.N*self.K, kernel_size=(self.height, self.width), groups=self.K)
            self.Wo = nn.Conv1d(self.N*self.K, self.K*num_classes, kernel_size=1, groups=self.K)
            self.Wg = nn.Conv1d(self.N*self.K, self.K, kernel_size=1, groups=self.K)
        ############################
        # the attention heads
        ############################
        ah = self.conv(x)
        ah = self.softmax(ah)
        # compute the loss (Equation 3. of paper)
        Mt = ah.repeat(1, self.K, 1, 1).transpose(-2, -1)
        M = ah.repeat(1, 1, self.K, 1).view(ah.size()[0], -1, ah.size()[2], ah.size()[3])
        loss = M.matmul(Mt).pow(2).sum(-1).sum(-1).sum(-1, keepdim=True) - ah.matmul(ah.transpose(-2, -1)).sum(-1).sum(-1).sum(-1, keepdim=True)
        ah = ah.repeat(1, 1, self.N, 1).view(ah.size()[0], -1, ah.size()[2], ah.size()[3])
        inp = x.repeat(1, self.K, 1, 1)
        ah_out = torch.mul(ah, inp)
        ############################
        # the output head
        ############################
        dim_red = self.dimred_conv(ah_out).squeeze(-1)
        hyp = self.Wo(dim_red).view(-1, self.K, self.num_classes)
        ############################
        # attention gate confidence
        ############################
        conf = self.tanh(self.Wg(dim_red))
        return hyp, conf, loss


class PainlessAttention(nn.Module):
    def __init__(self, model, num_heads, num_classes):
        super(PainlessAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-2)
        self.model = model
        self.pool_store = PoolStore()
        assert (hasattr(self.model, 'features') and hasattr(self.model, 'classifier')) or hasattr(self.model, 'fc'), 'Wrapped model must contain certain fields' 
        if hasattr(self.model, 'features'):
            for idx, module in self.model.features._modules.items():
                if 'Pool' in module.__class__.__name__:
                    ah = Attention(num_heads, num_classes)
                    self.model.features._modules[idx] = PoolCollector(module, ah, self.pool_store)
            # by assertion classifier also contained
            self.model.classifier = OutputCollector(self.model.classifier, self.pool_store)
        else:
            for idx, module in self.model._modules.items():
                if 'Pool' in module.__class__.__name__:
                    ah = Attention(num_heads, num_classes)
                    self.model._modules[idx] = PoolCollector(module, ah, self.pool_store)
            # by assertion fc also contained
            self.model.fc = OutputCollector(self.model.fc, self.pool_store)
    def forward(self, x):
        self.pool_store.reset()
        o = self.model(x)
        # concatenate outputs and confidences for all layers
        concat_hyps = torch.cat(self.pool_store.hyps + [o.unsqueeze(-2)], dim=-2)
        concat_confs = torch.cat(self.pool_store.confs + [self.pool_store.output_conf.unsqueeze(-2)], dim=-2)
        # compute combine output
        o = (self.softmax(concat_confs)*concat_hyps).sum(dim=-2)
        # compute loss
        loss = torch.cat(self.pool_store.losses).sum(-1)
        self.pool_store.reset()
        return o, loss


class PoolStore(nn.Module):
    def __init__(self):
        super(PoolStore, self).__init__()
        self.reset()
    def reset(self):
        self.results = []
        self.hyps = []
        self.confs = []
        self.losses = []
        self.output_conf = None
    def add_result(self, x, hyp, conf, loss):
        self.results.append(x)
        self.hyps.append(hyp)
        self.confs.append(conf)
        self.losses.append(loss)
    def add_output_conf(self, conf):
        self.output_conf = conf


class PoolCollector(nn.Module):
    def __init__(self, pool_layer, attention_head, pool_store):
        super(PoolCollector, self).__init__()
        self.pool_store = pool_store
        self.pool_layer = pool_layer
        self.attention_head = attention_head
    def forward(self, x):
        x = self.pool_layer(x)
        hyp, conf, loss = self.attention_head(x)
        self.pool_store.add_result(x, hyp, conf, loss)
        return x


class OutputCollector(nn.Module):
    def __init__(self, final_layer, pool_store):
        super(OutputCollector, self).__init__()
        self.pool_store = pool_store
        self.final_layer = final_layer
        self.conf_conv = None
    def forward(self, x):
        if self.conf_conv is None:
            self.conf_conv = nn.Linear(x.data.numpy().shape[1], 1)
        conf = self.conf_conv(x)
        o = self.final_layer(x)
        self.pool_store.add_output_conf(conf)
        return o


if __name__ == '__main__':
    num_classes = 7
    num_heads = 6
    num_imgs = 5
    num_channels = 3
    height = 20
    width = 20
    inp = torch.autograd.Variable(torch.rand((num_imgs, num_channels, height, width)), requires_grad=False)
    att = Attention(num_heads, num_classes)
    hyp, conf, loss = att(inp)
    print(hyp.data.numpy().shape)
    print(conf.data.numpy().shape)

    model = ExampleNet(num_classes)
    attn_model = PainlessAttention(model, num_heads, num_classes)
    out, loss = attn_model(inp)
    print(out.data.numpy().shape)
