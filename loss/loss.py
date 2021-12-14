import numpy as np
import torch
import torch.nn as nn
from common import se3, so3
from common.utils import tensor_gpu


class LossL1(nn.Module):
    def __init__(self):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super(LossCrossEntropy, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def __call__(self, input, target, weight=None):
        return self.loss(input, target)


def compute_loss(endpoints, params):
    loss = {}

    l1_criterion = LossL1()
    l2_criterion = LossL2()
    cls_criterion = LossCrossEntropy(weight=torch.tensor([0.7, 0.3]).cuda())
    num_iter = len(endpoints["all_pose_pair"])
    if params.loss_type == "omnet":
        for i in range(num_iter):
            # cls loss
            src_cls_pair, ref_cls_pair = endpoints['all_src_cls_pair'][i], endpoints['all_ref_cls_pair'][i]
            src_cls = cls_criterion(src_cls_pair[1], src_cls_pair[0].long())
            ref_cls = cls_criterion(ref_cls_pair[1], ref_cls_pair[0].long())
            loss['cls_{}'.format(i)] = (src_cls + ref_cls) / 2.0
            # reg loss
            pose_pair = endpoints["all_pose_pair"][i]
            loss["quat_{}".format(i)] = l1_criterion(pose_pair[0][:, :4], pose_pair[1][:, :4]) * params.loss_alpha1
            loss["translate_{}".format(i)] = l2_criterion(pose_pair[0][:, 4:], pose_pair[1][:, 4:]) * params.loss_alpha2
            # total loss
        total_loss = []
        for k in loss:
            total_loss.append(loss[k].float())
        loss["total"] = torch.sum(torch.stack(total_loss), dim=0)
    else:
        raise NotImplementedError

    return loss


def compute_metrics(endpoints, params):
    metrics = {}
    with torch.no_grad():
        gt_transforms = endpoints["transform_pair"][0]
        pred_transforms = endpoints["transform_pair"][1]

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        if "prnet" in params.transform_type:
            r_gt_euler_deg = so3.torch_dcm2euler(gt_transforms[:, :3, :3], seq="zyx")
            r_pred_euler_deg = so3.torch_dcm2euler(pred_transforms[:, :3, :3], seq="zyx")
        else:
            r_gt_euler_deg = so3.torch_dcm2euler(gt_transforms[:, :3, :3], seq="xyz")
            r_pred_euler_deg = so3.torch_dcm2euler(pred_transforms[:, :3, :3], seq="xyz")
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]

        r_mse = torch.mean((r_gt_euler_deg - r_pred_euler_deg)**2, dim=1)
        r_mae = torch.mean(torch.abs(r_gt_euler_deg - r_pred_euler_deg), dim=1)
        t_mse = torch.mean((t_gt - t_pred)**2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        r_mse = torch.mean(r_mse)
        t_mse = torch.mean(t_mse)
        r_mae = torch.mean(r_mae)
        t_mae = torch.mean(t_mae)

        # Rotation, translation errors (isotropic, i.e. doesn"t depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.torch_concatenate(se3.torch_inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)
        err_r = torch.mean(residual_rotdeg)
        err_t = torch.mean(residual_transmag)

        # weighted score of isotropic errors
        score = err_r * 0.01 + err_t

        metrics = {"R_MSE": r_mse, "R_MAE": r_mae, "t_MSE": t_mse, "t_MAE": t_mae, "Err_R": err_r, "Err_t": err_t, "score": score}
        metrics = tensor_gpu(metrics, check_on=False)

    return metrics
