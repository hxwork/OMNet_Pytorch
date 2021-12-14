import logging
import numpy as np
import torch
import torch.nn as nn
import transforms3d.euler as t3d
from common import quaternion, se3

from model.module import *


class OMNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.num_iter = params.titer
        self.encoder = nn.ModuleList([OMNetEncoder() for _ in range(self.num_iter)])
        self.fusion = nn.ModuleList([OMNetFusion() for _ in range(self.num_iter)])
        self.decoder = nn.ModuleList([OMNetDecoder() for _ in range(self.num_iter)])
        self.regression = nn.ModuleList([OMNetRegression() for _ in range(self.num_iter)])
        self.overlap_dist = params.overlap_dist

    def generate_overlap_mask(self, points_src: torch.Tensor, points_ref: torch.Tensor, mask_src: torch.Tensor, mask_ref: torch.Tensor,
                              transform_gt: torch.Tensor):
        points_src[torch.logical_not(mask_src), :] = 50.0
        points_ref[torch.logical_not(mask_ref), :] = 100.0
        points_src = se3.torch_transform(transform_gt, points_src)
        dist_matrix = torch.sqrt(torch.sum(torch.square(points_src[:, :, None, :] - points_ref[:, None, :, :]), dim=-1))  # (B, N, N)
        dist_s2r = torch.min(dist_matrix, dim=2)[0]
        dist_r2s = torch.min(dist_matrix, dim=1)[0]
        overlap_src_mask = dist_s2r < self.overlap_dist  # (B, N)
        overlap_ref_mask = dist_r2s < self.overlap_dist  # (B, N)

        return overlap_src_mask, overlap_ref_mask

    def forward(self, data):
        endpoints = {}

        xyz_src = data["points_src"][:, :, :3]
        xyz_ref = data["points_ref"][:, :, :3]
        transform_gt = data["transform_gt"]
        pose_gt = data["pose_gt"]

        # init endpoints
        all_src_cls_pair = []
        all_ref_cls_pair = []
        all_transform_pair = []
        all_pose_pair = []

        # init params
        B, src_N, _ = xyz_src.size()
        _, ref_N, _ = xyz_ref.size()
        init_quat = t3d.euler2quat(0., 0., 0., "sxyz")
        init_quat = torch.from_numpy(init_quat).expand(B, 4)
        init_translate = torch.from_numpy(np.array([[0., 0., 0.]])).expand(B, 3)
        pose_pred = torch.cat((init_quat, init_translate), dim=1).float().cuda()  # (B, 7)
        transform_pred = quaternion.torch_quat2mat(pose_pred)
        src_pred_mask = torch.ones(size=(B, src_N), dtype=xyz_src.dtype).cuda()
        ref_pred_mask = torch.ones(size=(B, ref_N), dtype=xyz_ref.dtype).cuda()
        overlap_src_mask, overlap_ref_mask = self.generate_overlap_mask(xyz_src.clone(), xyz_ref.clone(), src_pred_mask, ref_pred_mask,
                                                                        transform_gt)

        # rename xyz_src
        xyz_src_iter = xyz_src.clone()

        for i in range(self.num_iter):
            # mask deley
            if i < 2:
                src_pred_mask = torch.ones(size=(B, src_N), dtype=xyz_src.dtype).cuda()
                ref_pred_mask = torch.ones(size=(B, ref_N), dtype=xyz_ref.dtype).cuda()

            # encoder
            src_encoder_feats, src_glob_feat = self.encoder[i](xyz_src_iter.transpose(1, 2).detach(), src_pred_mask.unsqueeze(1))
            ref_encoder_feats, ref_glob_feat = self.encoder[i](xyz_ref.transpose(1, 2), ref_pred_mask.unsqueeze(1))

            # fusion
            src_cat_feat = torch.cat((src_encoder_feats[0], src_glob_feat.repeat(1, 1, src_N), ref_glob_feat.repeat(1, 1, src_N)), dim=1)
            ref_cat_feat = torch.cat((ref_encoder_feats[0], ref_glob_feat.repeat(1, 1, ref_N), src_glob_feat.repeat(1, 1, ref_N)), dim=1)
            _, src_fused_feat = self.fusion[i](src_cat_feat, src_pred_mask.unsqueeze(1))
            _, ref_fused_feat = self.fusion[i](ref_cat_feat, ref_pred_mask.unsqueeze(1))

            # decoder
            src_decoder_feats, src_cls_pred = self.decoder[i](src_fused_feat)
            ref_decoder_feats, ref_cls_pred = self.decoder[i](ref_fused_feat)

            # regression
            src_feat = torch.cat(src_decoder_feats, dim=1) * src_pred_mask.unsqueeze(1)
            ref_feat = torch.cat(ref_decoder_feats, dim=1) * ref_pred_mask.unsqueeze(1)
            cat_feat = torch.cat((src_fused_feat, src_feat, ref_fused_feat, ref_feat), dim=1)
            cat_feat = torch.max(cat_feat, dim=-1)[0]
            pose_pred_iter = self.regression[i](cat_feat)  # (B, 7)
            xyz_src_iter = quaternion.torch_quat_transform(pose_pred_iter, xyz_src_iter.detach())
            pose_pred = quaternion.torch_transform_pose(pose_pred.detach(), pose_pred_iter)
            transform_pred = quaternion.torch_quat2mat(pose_pred)

            # compute overlap and cls gt
            overlap_src_mask, overlap_ref_mask = self.generate_overlap_mask(xyz_src.clone(), xyz_ref.clone(), src_pred_mask, ref_pred_mask,
                                                                            transform_gt)
            src_cls_gt = torch.ones(B, src_N).cuda() * overlap_src_mask
            ref_cls_gt = torch.ones(B, ref_N).cuda() * overlap_ref_mask
            src_pred_mask = torch.argmax(src_cls_pred, dim=1)
            ref_pred_mask = torch.argmax(ref_cls_pred, dim=1)

            # add endpoints
            all_src_cls_pair.append([src_cls_gt, src_cls_pred])
            all_ref_cls_pair.append([ref_cls_gt, ref_cls_pred])
            all_transform_pair.append([transform_gt, transform_pred])
            all_pose_pair.append([pose_gt, pose_pred])

        endpoints["all_src_cls_pair"] = all_src_cls_pair
        endpoints["all_ref_cls_pair"] = all_ref_cls_pair
        endpoints["all_transform_pair"] = all_transform_pair
        endpoints["all_pose_pair"] = all_pose_pair
        endpoints["transform_pair"] = [transform_gt, transform_pred]
        endpoints["pose_pair"] = [pose_gt, pose_pred]

        return endpoints


def fetch_net(params):
    if params.net_type == "omnet":
        net = OMNet(params)

    else:
        raise NotImplementedError

    return net
