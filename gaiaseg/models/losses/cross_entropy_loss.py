import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import numpy as np

from ..builder import LOSSES
from .utils import weight_reduce_loss

def self_cross_entropy(pred,
                       label, # label 都还不是one-hot encoding 的形式
                       weight=None,
                       class_weight=None,
                       reduction='mean',
                       avg_factor=None,
                       ignore_index=255,
                       category_num=20
                       ):
    """The wrapper function for :func:softmax and nll loss"""
    # pdb.set_trace()
    N = label.shape[0]
    save_label = label
    label = torch.zeros(pred.shape).cuda()
    loss = torch.tensor(0.0).cuda()
    for i in range(N):
            if(torch.max(save_label[i])>255):
                # 说明要解码, 有没有更快的判断方式？
                # 都处理一下变成这种形式
                # tensor 都是浮点型的 好像还得单步调试的时候确认一下, 确认过了，不是浮点型，loadannotations的时候是什么类型就是什么类型
                # pdb.set_trace()
                temp_label = save_label[i] # 这个地方可以加个time函数测试下是用cpu解码快还是GPU解码快   
                
                for j in range(category_num): # 这个地方得加个参数，category_num
                    choice = temp_label&7 # 相当于取前三位 # 卧槽之前这里有bug，怪不得当时虽然当时用BCE，但是点数直掉
                    label[i,j,...][choice==1] = 0
                    label[i,j,...][choice==2] = 1
                    label[i,j,...][choice==4] = 255
                    temp_label = temp_label>>3
                # pdb.set_trace()
                temp_weight = torch.ones(pred[i].shape).cuda()
                temp_weight[label[i]==255] = 0
                temp_pred = pred[i].softmax(0)
                detach_pred = temp_pred.clone().detach()
                # 这里得用个新的变量
                new_pred = temp_pred*temp_weight + detach_pred*(1-temp_weight) # 这样确实比较巧妙。
                # 这个相当于算一个信息熵，这个和BCE loss相比最大的不同在于说，那个是sigmoid 得到
                # 某种程度上的概率值，这个则是softmax操作得到概率值。
                loss1 = torch.nn.functional.binary_cross_entropy(new_pred,label[i],reduction='sum')/torch.sum(1-temp_weight)
                loss = loss + loss1
                print("loss1: ",loss1)
            else:
                # 常规标签正常算loss
                # pdb.set_trace()
                loss2 =  F.cross_entropy(pred[i].unsqueeze(0),# 要求第一个维度是batch size
                                         save_label[i].unsqueeze(0),
                                         weight=class_weight,
                                         reduction='none',
                                         ignore_index=ignore_index)
                loss2 = weight_reduce_loss(
        loss2, weight=weight, reduction=reduction, avg_factor=avg_factor)
                print("loss2: ", loss2)
                loss = loss + loss2

    return loss

def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=255):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    # 这里的label还不是one-hot encoding 形式的，查了下F.cross_entropy就正好要求不是one-hot encoding
    # 的形式
    #  这个地方出现意见匪夷所思的事情，在cityscapes数据集有的pixel的label是30
    #  竟然在这里计算loss的时候没有报错，？？？
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights

def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=255,
                         category_num=20):
    """Calculate the binary CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255
    Returns:
        torch.Tensor: The calculated loss
    """
    # print("Starting Computing BCE Loss")
    # pdb.set_trace()
    weight = torch.ones(pred.shape).cuda()
    if pred.dim() != label.dim():
        N = label.shape[0]
        save_label = label
        label = torch.zeros(pred.shape).cuda()
        for i in range(N):
            if(torch.max(save_label[i])>255):
                # 说明要解码, 有没有更快的判断方式？
                # 都处理一下变成这种形式
                # tensor 都是浮点型的 好像还得单步调试的时候确认一下, 确认过了，不是浮点型，loadannotations的时候是什么类型就是什么类型
                temp_label = save_label[i] # 这个地方可以加个time函数测试下是用cpu解码快还是GPU解码快   
                # label_slice = torch.zeros(pred[0].shape).cuda()
                  
                for j in range(category_num): # 这个地方得加个参数，category_num
                    choice = temp_label&7 # 相当于取前三位
                    label[i,j,...][choice==1] = 0
                    label[i,j,...][choice==2] = 1
                    label[i,j,...][choice==4] = 255
                    temp_label = temp_label>>3
            else:
                temp_label = save_label[i].unsqueeze(0)
                temp_pred = pred[0].unsqueeze(0)
                assert (temp_pred.dim() == 2 and temp_label[i].dim() == 1) or (
                        temp_pred.dim() == 4 and temp_label.dim() == 3), \
                        'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
                        'H, W], label shape [N, H, W] are supported'
                temp_label, temp_weight = _expand_onehot_labels(temp_label, None, temp_pred.shape,
                                                      ignore_index)
                label[i] = temp_label.squeeze(0)
                weight[i] = temp_weight.squeeze(0)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    # pdb.set_trace()
    class_weight = torch.ones(label.shape).cuda()
    class_weight[label==255] = 0
    label[label==255] = 0
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)
 #   print("loss: ",loss)
    return loss

def equalize_loss(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=255):
    """Calculate the binary CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    '''
    这一段暂时用不到
    # backgroud percentage
    background_percent = torch.sum(label==0)
    person_percent = torch.sum(label==1)
    sky_percent = torch.sum(label==2)
    vegetation_percent = torch.sum(label==3)
    # 这里可以有个对比试验，判断一下，是每个batch size上算一次这个percent，
    # 还是说直接按照全局的groud truth确定percent
    '''

    class_weight = torch.ones(pred.shape)

    temp_high = len(label.view(-1))
                                                        # 0.618 is a hyperparameter
    temp_sample_index = np.random.randint(low=0, high=temp_high, size=(int(temp_high*0.618)), dtype='l')
    # class_weight = torch.ones(pred.shape) # redundancy
    temp_label = label.view(-1).cpu()
    sample_index = torch.zeros(temp_label.shape)
    sample_index[temp_sample_index] = 1
    sample_index = (sample_index == 1)
    class_weight[:,1][(sample_index*(temp_label!=1)).view(label.shape)] = 0.0               
    class_weight[:,2][(sample_index*(temp_label!=2)).view(label.shape)] = 0.0
    class_weight[:,3][(sample_index*(temp_label!=3)).view(label.shape)] = 0.0
#    class_weight[:,4][(sample_index*(temp_label!=4)).view(label.shape)] = 0.0
#    class_weight[:,5][(sample_index*(temp_label!=5)).view(label.shape)] = 0.0

    class_weight = class_weight.cuda()
    # 现在就是考虑怎么具体实现让哪些地方weight为0
    # 现在一个初步的实现想法是首先产生一个permutation，然后permutation中的地方
    # 为true，加入true的概率是0.3，然后这个再和（没有人的地方作与，为true的时候停止对人的抑制）
    # 感觉这个方法可以利用torch的内部函数，不用使用循环，初步判断效率应该还可以。

    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)
    
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none')

    # pdb.set_trace() 
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """
    每太get到这个masks的loss是什么意思？ igonore区域的loss？
    Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    # 这个[None]是个什么操作？？
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            or softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 use_eql=False,
                 use_selfCE=False,
                 reduction='mean',
                 class_weight=None,
                 category_num=20,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()

        # assert (use_sigmoid is False) or (use_mask is False) or (use_eql is False)
        # assert (use_sigmoid is False) or (use_mask is False) or (use_eql is False)
        # Either one or no one
        assert int(use_sigmoid) + int(use_mask) + int(use_eql) + int(use_selfCE)< 2

        self.use_eql = use_eql
        self.use_sigmoid = use_sigmoid
        self.use_selfCE = use_selfCE
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.category_num = category_num

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        elif self.use_eql:
            self.cls_criterion = equalize_loss
        elif self.use_selfCE:
            self.cls_criterion = self_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        #pdb.set_trace()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        pdb.set_trace()
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            category_num=self.category_num,
            **kwargs)
        return loss_cls
