import torch
import torch.nn.functional as F
class SmoothCrossEntropyLoss(torch.nn.KLDivLoss):
    def __init__(self, eps=0.1,ig_label =255, *args, **kargs):
        super(SmoothCrossEntropyLoss, self).__init__(*args, **kargs)
        self.eps = eps
        self.ignore_index = ig_label
    def forward(self, x, target):
        k = x.size(1)
        eps = self.eps / (k - 1)
        target_logit = torch.zeros((target.size(0), k, *(target.size()[1:])), 
                                   device=x.device).fill_(eps)
        target_fillzero = target*(target!=self.ignore_index).long()
        target_logit.scatter_(1, target_fillzero.unsqueeze(1), 1 - self.eps)
        #print (target_logit.size(),target_logit[:,0].size())
        #print (target_logit[:,0].size())
        target_logit[:,0][(target==self.ignore_index)] = eps
        if self.reduction=='elementwise_mean':
            return super(SmoothCrossEntropyLoss, self).forward(
                    F.log_softmax(x, 1), target_logit)*k
        elif self.reduction=='none':
            return super(SmoothCrossEntropyLoss, self).forward(
                    F.log_softmax(x, 1), target_logit).sum(1)
        elif self.reduction=='sum':
            return super(SmoothCrossEntropyLoss, self).forward(
                    F.log_softmax(x, 1), target_logit)
        else:
            assert (1<0)



if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ylabel = torch.tensor([[[0, 255]], [[3, 4]]], device=device)
    yprob = torch.tensor(
        [[[[0., 0.]],
         [[1., 0.]],
         [[0., 1.]],
         [[0., 0.]],
         [[0., 0.]]],
        [[[0., 0.]],
         [[0., 0.]],
         [[0., 0.]],
         [[1., 0.]],
         [[0., 1.]]]]
    , device=device)
    
    x = torch.tensor(
        [[[[0., 0.]],
         [[1, 0.]],
         [[1, 1.]],
         [[0., 0.]],
         [[0., 0.]]],
        [[[0., 0.]],
         [[0., 0.3]],
         [[0., 0.]],
         [[1., 0.]],
         [[0., 2]]]]
    , device=device)
    #print(torch.nn.CrossEntropyLoss(ignore_index=255)(x, ylabel))
    #print(F.cross_entropy(x, ylabel))
    #xlogit = F.log_softmax(x, 1)
    #print(torch.nn.functional.nll_loss(xlogit, ylabel))
    #print(F.kl_div(xlogit, yprob) * 5)
    #print(SmoothCrossEntropyLoss(eps=0)(x, ylabel))
    #print('-----------')
    #print(SmoothCrossEntropyLoss(eps=0.01)(x, ylabel))
    #print(SmoothCrossEntropyLoss(eps=0.001)(x, ylabel))
    #print('-----------')
    print(torch.nn.CrossEntropyLoss(ignore_index=255,reduction='elementwise_mean')(x, ylabel))
    print(SmoothCrossEntropyLoss(eps=0.0, reduction='elementwise_mean')(x, ylabel))
    print(torch.nn.CrossEntropyLoss(ignore_index=255,reduction='none')(x, ylabel).mean())
    print('-----------')
    print(torch.nn.CrossEntropyLoss(ignore_index=255,reduction='none')(x, ylabel))
    print(SmoothCrossEntropyLoss(eps=0.0, reduction='none')(x, ylabel))
    print('-----------')
    print(torch.nn.CrossEntropyLoss(ignore_index=255,reduction='sum')(x, ylabel))
    print(SmoothCrossEntropyLoss(eps=0.0, reduction='sum')(x, ylabel))