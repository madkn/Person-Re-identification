import torch
from torch.autograd import Variable

class LSROLoss(torch.nn.Module):
    def __init__(self, num_cl):
        super(LSROLoss, self).__init__()
        self.num_cl = num_cl
        
    def forward(self, batch_features, target, is_real):
        xmax = torch.max(batch_features, 1)[0].view(-1, 1)
        return torch.mean(
            is_real.view(-1, 1) * (- torch.gather(batch_features, 1, target.view(-1, 1)) + xmax)
            + (-1 / self.num_cl * torch.sum((1 - is_real).view(-1, 1) * (batch_features - xmax), 1))
            + (torch.log(torch.sum(torch.exp(batch_features - xmax), 1))).view(-1, 1)
        )
    
class HistogramLoss(torch.nn.Module):
    def __init__(self, num_steps):
        super(HistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.t = torch.range(-1, 1, self.step).view(-1, 1).cuda()
        self.tsize = self.t.size()[0]
        
    def forward(self, features, classes):
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (delta_repeat == (self.t - self.step)) & inds
            indsb = (delta_repeat == self.t) & inds
            s_repeat_[~(indsb|indsa)] = 0
            s_repeat_[indsa] = (s_repeat_ - Variable(self.t) + self.step)[indsa] / self.step
            s_repeat_[indsb] =  (-s_repeat_ + Variable(self.t) + self.step)[indsb] / self.step
            
            return s_repeat_.sum(1) / size
        
        classes_size = classes.size()[0]
        classes_eq = (classes.repeat(classes_size, 1)  == classes.view(-1, 1).repeat(1, classes_size)).data
        dists = torch.mm(features, features.transpose(0, 1))
        s_inds = torch.triu(torch.ones(dists.size()), 1).byte().cuda()
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum()
        neg_size = (~classes_eq[s_inds]).sum()
        s = dists[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        delta_repeat = (torch.floor((s_repeat.data + 1) / self.step) * self.step - 1).float()
        
        h_pos = histogram(pos_inds, pos_size)
        h_neg = histogram(neg_inds, neg_size)
        h_pos_repeat = h_pos.view(-1, 1).repeat(1, h_pos.size()[0])
        h_pos_repeat[torch.tril(torch.ones(h_pos_repeat.size()), -1).byte().cuda()] = 0
        h_pos_cdf = h_pos_repeat.sum(0)
        loss = torch.sum(h_neg * h_pos_cdf)
        
        return loss
    
