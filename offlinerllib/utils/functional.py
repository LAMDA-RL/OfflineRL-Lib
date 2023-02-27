import torch


def grad_gumbel(x, clip_max=7):
    x = torch.clamp(x, None, clip_max)
    x_max = torch.max(x, dim=0)
    x_max = torch.where(x_max < -1, -1, x_max)
    
    x1 = x - x_max
    return (torch.exp(x1) - torch.exp(-x_max)) / (torch.exp(x1) - x*torch.exp(-x_max)).mean(dim=0, keepdim=True)

def gumbel_log_loss(pred, target, alpha=1.0, clip_max=7.0):
    """
    use analytical form to improve stability, see https://github.com/Div99/XQL/blob/c54df7dd5bdbdd9af6ddead5887f13a23a54e535/offline/critic.py#L23
    
    """
    diff = (target - pred) / alpha
    return grad_gumbel(diff, clip_max=clip_max).detach() * diff

def gumbel_rescale_loss(pred, target, alpha=1.0, clip_max=7.0):
    diff = (target - pred) / alpha
    diff = diff.clamp(max=clip_max)
    max_diff = torch.max(diff, dim=0)[0]
    max_diff = torch.where(max_diff < -1.0, -1.0, max_diff)
    max_diff = max_diff.detach()
    # return torch.exp(-max_diff) * (torch.exp(diff)-diff-1)
    return torch.exp(diff - max_diff) - diff*torch.exp(-max_diff) - torch.exp(-max_diff)
    
def expectile_regression(pred, target, expectile):
    diff = target - pred
    return torch.where(diff > 0, expectile, 1-expectile) * (diff**2)

def discounted_cum_sum(seq, discount):
    seq = seq.copy()
    for t in reversed(range(len(seq)-1)):
        seq[t] += discount * seq[t+1]
    return seq