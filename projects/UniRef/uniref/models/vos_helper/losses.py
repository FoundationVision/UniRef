"""Semantic segmentation losses.
Copy-paste from TransVOS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    r'''
        pred: [T x no x H x W], including bg
        mask: [T x no x H x W] one-hot encoded
        bootstrap: float(default: 0.4)
    '''
    def __init__(self, bootstrap=0.4):
        super(CELoss, self).__init__()

        self.bootstrap = bootstrap

    def forward(self, predict, target):
        N, _, H, W = target.shape

        predict = -1 * torch.log(predict)

        # bootstrap
        if self.bootstrap > 0:
            num = int(H * W * self.bootstrap)

            loss = torch.sum(predict * target, dim=1).view(N, -1)
            mloss, _ = torch.sort(loss, dim=-1, descending=True)
            loss = torch.mean(mloss[:, :num])
        else:
            loss = torch.sum(predict * target)
            loss = loss / (H * W * N)

        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class MaskIouLoss(nn.Module):
    r'''
        pred: [N x K x H x W]
        mask: [N x K x H x W] one-hot encoded
        num_object: int
    '''
    def __init__(self,):
        super(MaskIouLoss, self).__init__()

    def mask_iou(self, predict, target, eps=1e-7, size_average=True):
        r"""
            param: 
                pred: size [N x H x W]
                target: size [N x H x W]
            output:
                iou: size [1] (size_average=True) or [N] (size_average=False)
        """
        assert len(predict.shape) == 3 and predict.shape == target.shape

        N = predict.size(0)

        inter = torch.min(predict, target).sum(2).sum(1)
        union = torch.max(predict, target).sum(2).sum(1)

        if size_average:
            iou = torch.sum((inter+eps) / (union+eps)) / N
        else:
            iou = (inter+eps) / (union+eps)

        return iou

    def forward(self, predict, target):
        N, _, _, _ = target.shape
        loss = torch.zeros(1).to(predict.device)

        for i in range(N):
            loss += (1.0 - self.mask_iou(predict[i], target[i]))

        loss = loss / N
        return loss


class SegLoss(nn.Module):
    r'''
        preds: [N, T, no, H, W]
        masks: [N, T, no, H, W]
        n_objs: [N]
    '''
    def __init__(self, alpha=1):
        super(SegLoss, self).__init__()
        self.alpha = alpha
        self.cls_loss = CELoss(bootstrap=0.4)
        self.iou_loss = MaskIouLoss()

    def forward(self, preds, masks, n_objs):
        # preds: [B x T-1 x M x H x W], scores
        # masks: [B x T-1 x M x H x W], each dimension in M is value in [0,1]
        # M including bg
        N = preds.shape[0]

        cls_loss = 0.0
        iou_loss = 0.0
        for i in range(N):
            n_obj = n_objs[i] # including bg
            cls_loss += self.cls_loss(preds[i, :, :n_obj], masks[i, :, :n_obj])
            iou_loss += self.iou_loss(preds[i, :, 1:n_obj], masks[i, :, 1:n_obj]) # start from 1 since 0 is background
        loss = cls_loss + self.alpha * iou_loss
        cls_loss /= N
        iou_loss /= N
        loss /= N
        loss_stats = {
            'loss_mask_cls': cls_loss,
            'loss_mask_iou': iou_loss
        }
        return loss, loss_stats
