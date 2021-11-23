# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch

import utils.misc as utils
import utils.box_ops as box_ops

from clip import clip
from clip.model import convert_weights
from utils.swig_evaluator import SWiGEvaluator
from utils.hico_evaluator import HICOEvaluator
from utils.visualize import visualize_preds
from cascade_hoist.layers import batched_nms
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


def prepare_inputs(images, targets, device):
    """Prepare model inputs."""
    images = images.to(device)
    targets = [{k: v.to(device) if k != "hois" else v for k, v in t.items()} for t in targets]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    texts = []
    text_inputs = []
    unique_hois = set()
    for t in targets:
        for hoi in t["hois"]:
            # Ensure all texts are unique (no duplicates).
            hoi_id = hoi["hoi_id"]
            if hoi_id in unique_hois:
                continue
            else:
                unique_hois.add(hoi_id)
            action_text, object_text = hoi["text"]
            action_token = _tokenizer.encode(action_text)
            object_token = _tokenizer.encode(object_text)
            
            action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
            object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
            texts.append([action_token, object_token])
            text_inputs.append(action_text + " " + object_text)
    
    return images, targets, texts


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for images, targets in metric_logger.log_every(data_loader, print_freq, header): 
               
        images, targets, texts = prepare_inputs(images, targets, device)
        outputs = model(images.tensors, texts, images.mask)
        loss_dict, indices = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # visualize_preds(images, targets, outputs, indices)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    convert_weights(model)

    if args.dataset_file == "swig":
        anno_file = "/raid1/suchen/repo/promting_hoi/data/swig_hoi/swig_dev_1000.json"
        evaluator = SWiGEvaluator(anno_file, args.output_dir)
    if args.dataset_file == "hico":
        anno_file = "/raid1/suchen/repo/promting_hoi/data/HICO-DET/test_hico_ann.json"
        evaluator = HICOEvaluator(anno_file, args.output_dir)
    else:
        raise NotImplementedError

    text_features = prepare_text_inputs(model, data_loader.dataset.dataset_texts, device)
    
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device)
        targets = [{k: v.to(device) if k != "hois" else v for k, v in t.items()} for t in targets]

        _, hoi_features, box_features, conf_features, attn_maps = model.encode_image(images.tensors, images.mask)

        hoi_features = hoi_features / hoi_features.norm(dim=-1, keepdim=True)
        pred_boxes = model.bbox_embed(box_features).sigmoid().float()
        conf_scores = model.hoi_confidence_embed(F.relu(conf_features))
        logits_per_hoi = model.logit_scale.exp() * hoi_features @ text_features.t()
        
        outputs = {"logits_per_hoi": logits_per_hoi, "pred_boxes": pred_boxes, "confidence_scores": conf_scores, "attention_maps": attn_maps}
        
        loss_dict, indices = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # visualize_preds(images, targets, outputs, indices)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        results = {int(targets[i]["image_id"]):
            postprocessing(
                logits_per_hoi[i],
                conf_scores[i],
                pred_boxes[i],
                images.mask[i],
                targets[i]["orig_size"],
                data_loader.dataset.text_mapper,
                args.test_score_thresh,
            ) for i in range(len(images.mask))
        }
        
        evaluator.update(results)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluator.save_preds()
    # accumulate predictions from all images
    evaluator.accumulate()
    evaluator.summarize()
    evaluator.save()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, evaluator


@torch.no_grad()
def prepare_text_inputs(model, texts, device):
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    
    text_tokens = []
    for action_text, object_text in texts:
        action_token = _tokenizer.encode(action_text)
        object_token = _tokenizer.encode(object_text)
        
        action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
        object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
        text_tokens.append([action_token, object_token])

    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def postprocessing(hoi_scores, conf_scores, pred_boxes, img_mask, orig_size, indices_mapper, test_thresh=0.001):
    hoi_scores = hoi_scores.softmax(dim=-1)
    conf_scores = conf_scores.sigmoid()
    scores = hoi_scores * conf_scores
    
    keep = torch.nonzero(scores > test_thresh, as_tuple=True)
    
    pred_person_boxes, pred_object_boxes = postprocessing_boxes(pred_boxes, img_mask, orig_size)

    # Apply nms
    scores = scores[keep]
    classes = keep[1]
    pred_person_boxes = pred_person_boxes[keep[0]]
    pred_object_boxes = pred_object_boxes[keep[0]]
    
    person_keep = batched_nms(pred_person_boxes, scores, classes, 0.5)
    object_keep = batched_nms(pred_object_boxes, scores, classes, 0.5)
    
    person_filter_mask = torch.zeros_like(scores, dtype=torch.bool)
    object_filter_mask = torch.zeros_like(scores, dtype=torch.bool)
    person_filter_mask[person_keep] = True
    object_filter_mask[object_keep] = True
    filter_mask = torch.logical_or(person_filter_mask, object_filter_mask)
    
    scores = scores[filter_mask].detach().cpu().numpy().tolist()
    classes = classes[filter_mask].detach().cpu().numpy().tolist()
    pred_boxes = torch.cat([pred_person_boxes, pred_object_boxes], dim=-1)
    pred_boxes = pred_boxes[filter_mask].detach().cpu().numpy().tolist()
    
    results = []
    for score, hoi_id, boxes in zip(scores, classes, pred_boxes):
        results.append([indices_mapper[int(hoi_id)], score] + boxes)

    return results


def postprocessing_boxes(pred_boxes, img_mask, orig_size):
    # Map the pred boxes to the original image size
    pred_person_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes[:, :4])
    pred_object_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes[:, 4:])
    
    pred_person_boxes = pred_person_boxes.clamp(min=0, max=1)
    pred_object_boxes = pred_object_boxes.clamp(min=0, max=1)
    
    h, w = img_mask.shape
    pred_person_boxes[:, 0::2] = pred_person_boxes[:, 0::2] * w
    pred_person_boxes[:, 1::2] = pred_person_boxes[:, 1::2] * h
    pred_object_boxes[:, 0::2] = pred_object_boxes[:, 0::2] * w
    pred_object_boxes[:, 1::2] = pred_object_boxes[:, 1::2] * h
    
    img_mask = ~img_mask
    valid_min_x = int(torch.nonzero(img_mask, as_tuple=True)[1].min())
    valid_min_y = int(torch.nonzero(img_mask, as_tuple=True)[0].min())
    
    pred_person_boxes[:, 0::2] -= valid_min_x
    pred_person_boxes[:, 1::2] -= valid_min_y
    pred_object_boxes[:, 0::2] -= valid_min_x
    pred_object_boxes[:, 1::2] -= valid_min_y

    pred_person_boxes[:, 0::2] = pred_person_boxes[:, 0::2].clamp(min=0, max=w-2*valid_min_x)
    pred_person_boxes[:, 1::2] = pred_person_boxes[:, 1::2].clamp(min=0, max=h-2*valid_min_y)
    pred_object_boxes[:, 0::2] = pred_object_boxes[:, 0::2].clamp(min=0, max=w-2*valid_min_x)
    pred_object_boxes[:, 1::2] = pred_object_boxes[:, 1::2].clamp(min=0, max=h-2*valid_min_y)

    ori_h, ori_w = orig_size
    scale_x, scale_y = ori_w / w, ori_h / h
    ratio = max(scale_x, scale_y)
    pred_person_boxes[:, 0::2] *= ratio
    pred_person_boxes[:, 1::2] *= ratio
    pred_object_boxes[:, 0::2] *= ratio
    pred_object_boxes[:, 1::2] *= ratio
    return pred_person_boxes, pred_object_boxes