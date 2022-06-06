# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Suchen for HOI detection
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_conf: float = 1,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_conf = cost_conf
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_conf != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits_per_hoi": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 "box_scores": Tensor of dim [batch_size, num_queries, 1] with the predicted box confidence scores

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["logits_per_hoi"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits_per_hoi"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 8]
        out_conf = outputs["box_scores"].flatten(0, 1).sigmoid() # [batch_size * num_queries, 1]

        # Also concat the target labels and boxes. During the training, due to the limit
        # GPU memory, we also consider the texts within each mini-batch. Differently, during
        # the inference, we consider all interactions in the dataset.
        unique_hois, cnt = {}, 0
        tgt_ids = []
        for t in targets:
            for hoi in t["hois"]:
                hoi_id = hoi["hoi_id"]
                if self.training:
                    # Only consider the texts within each mini-batch
                    if hoi_id not in unique_hois:
                        unique_hois[hoi_id] = cnt
                        cnt += 1
                    tgt_ids.append(unique_hois[hoi_id])
                else:
                    # Consider all hois in the dataset
                    tgt_ids.append(hoi_id)
        tgt_ids = torch.as_tensor(tgt_ids, dtype=torch.int64, device=out_prob.device)

        tgt_bbox = [torch.cat([t["boxes"][hoi["subject_id"]], t["boxes"][hoi["object_id"]]])
                    for t in targets for hoi in t["hois"]]
        tgt_bbox = torch.stack(tgt_bbox, dim=0)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the confidence cost
        cost_conf = -out_conf

        # Compute the L1 cost between boxes
        if out_bbox.dtype == torch.float16:
            out_bbox = out_bbox.type(torch.float32)
        cost_pbbox = torch.cdist(out_bbox[:, :4], tgt_bbox[:, :4], p=1)
        cost_obbox = torch.cdist(out_bbox[:, 4:], tgt_bbox[:, 4:], p=1)

        # Compute the giou cost betwen boxes
        cost_pgiou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[:, :4]), box_cxcywh_to_xyxy(tgt_bbox[:, :4]))
        cost_ogiou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[:, 4:]), box_cxcywh_to_xyxy(tgt_bbox[:, 4:]))

        # Final cost matrix
        C = self.cost_bbox * cost_pbbox + self.cost_bbox * cost_obbox + \
            self.cost_giou * cost_pgiou + self.cost_giou * cost_ogiou + \
            self.cost_class * cost_class + self.cost_conf * cost_conf
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["hois"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_conf=args.set_cost_conf,
    )