# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Suchen for HOI detection
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import torch
import utils.misc as utils
from models.model import convert_weights
from datasets import build_evaluator
from utils.visualizer import Visualizer
from fvcore.nn import FlopCountAnalysis, flop_count_table
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


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

        images, targets, texts = prepare_inputs(images, targets, data_loader, device)
        outputs = model(images.tensors, texts, images.mask)
        loss_dict, indices = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
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
def evaluate(model, postprocessors, criterion, data_loader, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # Convert applicable model parameters to fp16
    convert_weights(model)

    # Build evaluator
    evaluator = build_evaluator(args)

    # Convert all interaction categories into embeddings
    text_features = prepare_text_inputs(model, data_loader.dataset.dataset_texts, device)

    # Inference
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device)
        targets = [{k: v.to(device) if k != "hois" else v for k, v in t.items()} for t in targets]

        vision_outputs = model.encode_image(images.tensors, images.mask)

        hoi_features = vision_outputs['hoi_features']
        hoi_features = hoi_features / hoi_features.norm(dim=-1, keepdim=True)
        logits_per_hoi = model.logit_scale.exp() * hoi_features @ text_features.t()
        pred_boxes = vision_outputs["pred_boxes"]
        box_scores = vision_outputs["box_scores"]

        outputs = {"logits_per_hoi": logits_per_hoi,
                   "pred_boxes": pred_boxes,
                   "box_scores": box_scores,
                   "aux_outputs": vision_outputs["aux_outputs"],
                   "attn_maps": vision_outputs['attn_maps']}

        loss_dict, indices = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        if args.vis_outputs:
            visualizer = Visualizer(args)
            visualizer.visualize_preds(images, targets, outputs)
            # visualizer.visualize_attention(images, targets, outputs)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        results = {int(targets[i]['image_id']): postprocessors(
            {'pred_logits': logits_per_hoi[i], 'pred_boxes': pred_boxes[i], 'box_scores': box_scores[i]},
            targets[i]['orig_size'],
            data_loader.dataset.text_mapper
        ) for i in range(len(images.tensors))}

        evaluator.update(results)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluator.save_preds()
    # accumulate predictions from all images
    evaluator.accumulate()
    evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, evaluator


def prepare_inputs(images, targets, data_loader, device):
    """Prepare model inputs."""
    # image inputs
    images = images.to(device)
    targets = [{k: v.to(device) if k != "hois" else v for k, v in t.items()} for t in targets]

    # text inputs
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
            action_token = _tokenizer.encode(action_text.replace("_", " "))
            object_token = _tokenizer.encode(object_text.replace("_", " "))

            action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
            object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
            texts.append([action_token, object_token])
            text_inputs.append(action_text + " " + object_text)

    # [specific for HICO-DET], load related hois based on the targets in mini-batch
    if hasattr(data_loader.dataset, 'object_to_related_hois') and hasattr(data_loader.dataset, 'action_to_related_hois'):
        object_to_related_hois = data_loader.dataset.object_to_related_hois
        action_to_related_hois = data_loader.dataset.action_to_related_hois

        related_texts = []
        related_text_inputs = []
        unique_actions = set()
        unique_objects = set()
        unique_related_hois = set()
        for t in targets:
            for hoi in t["hois"]:
                hoi_id = hoi["hoi_id"]
                query_action_text, query_object_text = hoi["text"]
                if query_action_text in unique_actions or query_object_text in unique_objects:
                    continue
                else:
                    unique_actions.add(query_action_text)
                    unique_objects.add(query_object_text)

                related_hois = action_to_related_hois[query_action_text]
                for hoi in related_hois:
                    hoi_id = hoi["hoi_id"]
                    if hoi_id in unique_hois:
                        continue
                    if hoi_id in unique_related_hois:
                        continue
                    else:
                        unique_related_hois.add(hoi_id)

                    action_text, object_text = hoi["text"]
                    action_token = _tokenizer.encode(action_text.replace("_", " "))
                    object_token = _tokenizer.encode(object_text.replace("_", " "))
                    action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
                    object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
                    related_texts.append([action_token, object_token])
                    related_text_inputs.append(action_text + " " + object_text)

                related_hois = object_to_related_hois[query_object_text]
                for hoi in related_hois:
                    hoi_id = hoi["hoi_id"]
                    if hoi_id in unique_hois:
                        continue
                    if hoi_id in unique_related_hois:
                        continue
                    else:
                        unique_related_hois.add(hoi_id)

                    action_text, object_text = hoi["text"]
                    action_token = _tokenizer.encode(action_text.replace("_", " "))
                    object_token = _tokenizer.encode(object_text.replace("_", " "))
                    action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
                    object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
                    related_texts.append([action_token, object_token])
                    related_text_inputs.append(action_text + " " + object_text)
        texts.extend(related_texts)

    return images, targets, texts


@torch.no_grad()
def prepare_text_inputs(model, texts, device):
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    text_tokens = []
    for action_text, object_text in texts:
        action_token = _tokenizer.encode(action_text.replace("_", " "))
        object_token = _tokenizer.encode(object_text.replace("_", " "))

        action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
        object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
        text_tokens.append([action_token, object_token])

    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_flop_stats(model, data_loader):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.
    Returns:
        float: the total number of gflops of the given model.
    """
    inputs = _get_model_analysis_input(data_loader)
    flops = FlopCountAnalysis(model, inputs)
    print("Total FLOPs(G)", flops.total() / 1e9)
    print(flop_count_table(flops, max_depth=4, show_param_shapes=False))
    return flops


def _get_model_analysis_input(data_loader):
    for images, targets in data_loader:
        images, targets, texts = prepare_inputs(images, targets, "cuda")
        inputs = (images.tensors, texts, images.mask)
        return inputs


''' deprecated, text
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
            action_token = _tokenizer.encode(action_text.replace("_", " "))
            object_token = _tokenizer.encode(object_text.replace("_", " "))

            action_token = torch.as_tensor([sot_token] + action_token, dtype=torch.long).to(device)
            object_token = torch.as_tensor(object_token + [eot_token], dtype=torch.long).to(device)
            texts.append([action_token, object_token])
            text_inputs.append(action_text + " " + object_text)

    return images, targets, texts
'''