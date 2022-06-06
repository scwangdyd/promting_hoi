import collections
import numpy as np
import json
import os
import pickle
from .hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS
from .hico_categories import ZERO_SHOT_INTERACTION_IDS, NON_INTERACTION_IDS


class HICOEvaluator(object):
    ''' Evaluator for HICO-DET dataset '''
    def __init__(self, anno_file, output_dir):
        size = 600
        self.size = size
        self.gts = self.load_anno(anno_file)
        self.scores = {i: [] for i in range(size)}
        self.boxes = {i: [] for i in range(size)}
        self.keys = {i: [] for i in range(size)}
        self.hico_ap  = np.zeros(size)
        self.hico_rec = np.zeros(size)
        self.output_dir = output_dir

    def update(self, predictions):
        ''' Store predictions
        Args:
            predictions (dict): a dictionary in the following format.
            {
                img_id: [
                    [hoi_id, score, pbox_x1, pbox_y1, pbox_x2, pbox_y2, obox_x1, obox_y1, obox_x2, obox_y2],
                    ...
                    ...
                ]
            }
        '''
        for img_id, preds in predictions.items():
            for pred in preds:
                hoi_id = pred[0]
                score = pred[1]
                boxes = pred[2:]
                self.scores[hoi_id].append(score)
                self.boxes[hoi_id].append(boxes)
                self.keys[hoi_id].append(img_id)

    def accumulate(self):
        for hoi_id in range(600):
            gts_per_hoi = self.gts[hoi_id]
            ap, rec = calc_ap(self.scores[hoi_id], self.boxes[hoi_id], self.keys[hoi_id], gts_per_hoi)
            self.hico_ap[hoi_id], self.hico_rec[hoi_id] = ap, rec

    def summarize(self):
        valid_hois = np.setdiff1d(np.arange(600), NON_INTERACTION_IDS)
        seen_hois = np.setdiff1d(valid_hois, ZERO_SHOT_INTERACTION_IDS)
        zero_shot_hois = np.setdiff1d(ZERO_SHOT_INTERACTION_IDS, NON_INTERACTION_IDS)
        zero_shot_mAP = np.mean(self.hico_ap[zero_shot_hois])
        seen_mAP = np.mean(self.hico_ap[seen_hois])
        print("zero-shot mAP: {:.2f}".format(zero_shot_mAP * 100.))
        print("seen mAP: {:.2f}".format(seen_mAP * 100.))
        print("full mAP: {:.2f}".format(np.mean(self.hico_ap[valid_hois]) * 100.))

    def save_preds(self):
        with open(os.path.join(self.output_dir, "preds.pkl"), "wb") as f:
            pickle.dump({"scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def save(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        with open(os.path.join(output_dir, "dets.pkl"), "wb") as f:
            pickle.dump({"gts": self.gts, "scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def load_anno(self, anno_file):
        with open(anno_file, "r") as f:
            dataset_dicts = json.load(f)

        action_id2name = {x["id"]: x["name"] for x in HICO_ACTIONS}
        object_id2name = {x["id"]: x["name"] for x in HICO_OBJECTS}
        hoi_mapper = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

        size = self.size
        gts = {i: collections.defaultdict(list) for i in range(size)}
        for anno_dict in dataset_dicts:
            image_id = anno_dict["img_id"]
            box_annos = anno_dict.get("annotations", [])
            hoi_annos = anno_dict.get("hoi_annotation", [])
            for hoi in hoi_annos:
                person_box = box_annos[hoi["subject_id"]]["bbox"]
                object_box = box_annos[hoi["object_id"]]["bbox"]
                action_id = hoi["category_id"] - 1 # original annotations start from 1
                object_id = box_annos[hoi["object_id"]]["category_id"] # original annotations start from 1
                hoi_id = hoi_mapper[(action_id2name[action_id], object_id2name[object_id])]
                gts[hoi_id][image_id].append(person_box + object_box)

        for hoi_id in gts:
            for img_id in gts[hoi_id]:
                gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

        return gts


def calc_ap(scores, boxes, keys, gt_boxes):

    if len(keys) == 0:
        return 0, 0

    if isinstance(boxes, list):
        scores, boxes, key = np.array(scores), np.array(boxes), np.array(keys)

    hit = []
    idx = np.argsort(scores)[::-1]
    npos = 0
    used = {}

    for key in gt_boxes.keys():
        npos += gt_boxes[key].shape[0]
        used[key] = set()

    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        box = boxes[pair_id, :]
        key = keys[pair_id]
        if key in gt_boxes:
            maxi = 0.0
            k    = -1
            for i in range(gt_boxes[key].shape[0]):
                tmp = calc_hit(box, gt_boxes[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k    = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    rec    = hit / npos if npos > 0 else hit / (npos + 1e-8)
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec) if len(rec) else 0


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def iou(bb1, bb2, debug = False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)


''' deprecated, evaluator
def hico_evaluation(predictions, gts):
    images, results = [], []
    for img_key, ps in predictions.items():
        images.extend([img_key] * len(ps))
        results.extend(ps)

    hico_ap, hico_rec = np.zeros(600), np.zeros(600)

    scores = [[] for _ in range(600)]
    boxes = [[] for _ in range(600)]
    keys = [[] for _ in range(600)]

    for img_id, det in zip(images, results):
        hoi_id, person_box, object_box, score = int(det[0]), det[1], det[2], det[-1]
        scores[hoi_id].append(score)
        boxes[hoi_id].append([float(x) for x in person_box] + [float(x) for x in object_box])
        keys[hoi_id].append(img_id)

    for hoi_id in range(600):
        gts_per_hoi = gts[hoi_id]
        ap, rec = calc_ap(scores[hoi_id], boxes[hoi_id], keys[hoi_id], gts_per_hoi)
        hico_ap[hoi_id], hico_rec[hoi_id] = ap, rec

    return hico_ap, hico_rec


def prepare_hico_gts(anno_file):
    """
    Convert dataset to the format required by evaluator.
    """
    with open(anno_file, "r") as f:
        dataset_dicts = json.load(f)

    action_mapper = {x["name"]: x["id"]+1 for x in HICO_ACTIONS}
    object_mapper = {x["name"]: x["id"] for x in HICO_OBJECTS}
    hoi_mapper = {(action_mapper[x["action"]], object_mapper[x["object"]]): x["interaction_id"]
                  for x in HICO_INTERACTIONS}

    gts = {i: collections.defaultdict(list) for i in range(600)}
    for anno_dict in dataset_dicts:
        image_id = int(anno_dict["file_name"].split("_")[-1].split(".")[0])
        box_annos = anno_dict.get("annotations", [])
        hoi_annos = anno_dict.get("hoi_annotation", [])
        for hoi in hoi_annos:
            person_box = box_annos[hoi["subject_id"]]["bbox"]
            object_box = box_annos[hoi["object_id"]]["bbox"]
            action_id = hoi["category_id"]
            object_id = box_annos[hoi["object_id"]]["category_id"]
            hoi_id = hoi_mapper[(action_id, object_id)]
            gts[hoi_id][image_id].append(person_box + object_box)

    for hoi_id in gts:
        for img_id in gts[hoi_id]:
            gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

    return gts
'''