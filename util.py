import torch


class AverageMeter:
    """
    Computes and stores the average and current value of a given metric.

    From Mike Wu.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cp_model(src_path, tgt_path=None):
    model = torch.load(src_path)
    if tgt_path is None:
        src_split = src_path.split('/')
        tgt_path = '/'.join(src_split[:-1])[:-1] + '.pt'
    print(f'Model is copied to {tgt_path}')
    torch.save(model, tgt_path)
