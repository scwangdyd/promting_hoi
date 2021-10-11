"""
This script implements the baseline method: Disjoint object detector + Pretrained CLIP model.

Basic idea: we can learn an off-the-shelf object detector to first produce
the bounding boxes for all humans and objects. Then we build human-object pairs.
For each pair, we crop their union region and send it to the pretrained CLIP model.

This script assumes that the boxes have been computed (should be given as the input).
"""
import argparse
import os
import json
import pickle
import clip
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.hico_evaluator import hico_evaluation, prepare_hico_gts
from utils.hico_categories import (
    HICO_INTERACTIONS,
    HICO_OBJECTS,
    VERB_MAPPER,
    ZERO_SHOT_INTERACTION_IDS,
    NON_INTERACTION_IDS
)
from utils.swig_evaluator import swig_evaluation, prepare_swig_gts
from utils.swig_v1_categories import SWIG_ACTIONS, SWIG_CATEGORIES, SWIG_INTERACTIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="SWIG", type=str, choices=["HICO", "SWIG"],
                        help="Experiments on which dataset")
    parser.add_argument("--precomputed-boxes", type=str,
                        #default="/raid1/suchen/repo/baselines_hoi/DRG/Data/test_HICO_finetuned_v3.pkl",
                        default="/raid1/suchen/repo/promting_hoi/data/precomputed/swig_hoi/swig_dev_JSL_boxes.pkl",
                        help="path to the precomputed boxes.")
    parser.add_argument("--dataset-annos", type=str,
                        # default="/raid1/suchen/repo/promting_hoi/data/HICO-DET/test_hico.json",
                        default="/raid1/suchen/repo/promting_hoi/data/swig_hoi/swig_dev_1000.json",
                        help="path to the dataset annotations.")
    return parser.parse_args()


def load_precomputed_boxes(args):
    """ Load precomputed boxes from the given file (default in .pkl). """
    if args.exp == "HICO":
        img_dir = "/raid1/suchen/dataset/hico_20160224_det/images/test2015"
        
        with open(args.precomputed_boxes, "rb") as f:
            boxes = pickle.load(f)
        with open(args.dataset_annos, "r") as f:
            img_annos = json.load(f)
        
        id_to_filename = {}
        for img_dict in img_annos:
            img_id = int(img_dict["file_name"].split("_")[-1].split(".")[0])
            img_filename = os.path.join(img_dir, img_dict["file_name"])
            id_to_filename[img_id] = img_filename
        
        boxes_dict = {}
        for img_id, box in boxes.items():
            if img_id not in id_to_filename:
                continue
            img_filename = id_to_filename[img_id]
            boxes_dict[img_filename] = box
            
    elif args.exp == "SWIG":
        img_dir = "/raid1/suchen/dataset/swig/images_512/"
        
        with open(args.precomputed_boxes, "rb") as f:
            boxes = pickle.load(f)
            
        boxes_dict = {}
        for img_name, dets in boxes.items():
            img_filename = os.path.join(img_dir, img_name)
            boxes_dict[img_filename] = dets

    return boxes_dict


def build_ho_pairs(args, boxes):
    """ Pair every human and object box, and return the union region. """
    if args.exp == "HICO":
        person_boxes = []
        object_boxes = []
        for box_data in boxes:
            box_dict = {"box": box_data[2], "score": box_data[-1], "category_id": box_data[-2]}
            score = box_dict["score"]
            if score < 0.2:
                continue
            if box_data[1] == "Human":
                person_boxes.append(box_dict)
            else:
                object_boxes.append(box_dict)
        
        ho_pairs = []
        for person_dict in person_boxes:
            for object_dict in object_boxes:
                person_box = person_dict["box"]
                object_box = object_dict["box"]
                ul = [min(person_box[0], object_box[0]), min(person_box[1], object_box[1])]
                br = [max(person_box[2], object_box[2]), max(person_box[3], object_box[3])]
                ho_pairs.append({
                    "person_box": person_box,
                    "object_box": object_box,
                    "union_box": ul + br,
                    "person_score": person_dict["score"],
                    "object_score": object_dict["score"],
                    "object_category": object_dict["category_id"] - 1 # start from 1
                })
    
    elif args.exp == "SWIG":
        person_boxes = []
        object_boxes = []
        for box_data in boxes:
            box_dict = {"box": box_data[2:], "score": box_data[1], "category_id": box_data[0]}
            score = box_data[1]
            if score < 0.01:
                continue
            if box_data[0] == 0:
                person_boxes.append(box_dict)
            else:
                object_boxes.append(box_dict)
        
        ho_pairs = []
        for person_dict in person_boxes:
            for object_dict in object_boxes:
                person_box = person_dict["box"]
                object_box = object_dict["box"]
                ul = [min(person_box[0], object_box[0]), min(person_box[1], object_box[1])]
                br = [max(person_box[2], object_box[2]), max(person_box[3], object_box[3])]
                ho_pairs.append({
                    "person_box": person_box,
                    "object_box": object_box,
                    "union_box": ul + br,
                    "person_score": person_dict["score"],
                    "object_score": object_dict["score"],
                    "object_category": object_dict["category_id"]
                })

    return ho_pairs


def prepare_text_inputs(args, model):
    """ Encode the classes using pre-trained CLIP text encoder. """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.exp == "HICO":
        text_inputs = []
        indices_mapper = {}
        for i, hoi in enumerate(HICO_INTERACTIONS):
            act = hoi["action"]
            if act == "no_interaction":
                continue
            act = act.split("_")
            act[0] = VERB_MAPPER[act[0]]
            act = " ".join(act)
            obj = hoi["object"]
            s = f"a photo of people {act} {obj}."
            indices_mapper[len(text_inputs)] = i
            text_inputs.append(s)
    
    elif args.exp == "SWIG":
        text_inputs = []
        indices_mapper = {}
        text_freq = {}
        for i, hoi in enumerate(SWIG_INTERACTIONS):
            if hoi["evaluation"] == 0: continue
            action_id = hoi["action_id"]
            object_id = hoi["object_id"]
            
            act = SWIG_ACTIONS[action_id]["name"]
            obj = SWIG_CATEGORIES[object_id]["name"]
            act_def = SWIG_ACTIONS[action_id]["def"]
            obj_def = SWIG_CATEGORIES[object_id]["def"]
            obj_gloss = SWIG_CATEGORIES[object_id]["gloss"]
            obj_gloss = [obj] + [x for x in obj_gloss if x != obj]
            if len(obj_gloss) > 1:
                obj_gloss = " or ".join(obj_gloss)
            else:
                obj_gloss = obj_gloss[0]
            # s = f"A photo of a person {act} with object {obj}. The object {obj} means {obj_def}."
            # s = f"a photo of a person {act} with object {obj}"
            # s = f"A photo of a person {act} with {obj}. The {act} means to {act_def}."
            s = f"A photo of a person {act} with {obj_gloss}. The {act} means to {act_def}."
            indices_mapper[len(text_inputs)] = i
            text_freq[s] = hoi["frequency"]
            text_inputs.append(s)
    
    text_tokens = torch.cat([clip.tokenize(s) for s in text_inputs]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features, text_inputs, indices_mapper
     

def predict(args, model, preprocess, text_features, text_inputs, indices_mapper, img_filename, ho_pairs):
    """ Inference using pretrained CLIP model. """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image = Image.open(img_filename)
    
    predictions = []
    for ho_dict in ho_pairs:
        union_box = ho_dict["union_box"]
        cropped_image = image.crop(tuple(union_box))
        
        image_input = preprocess(cropped_image).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Filter out text inputs
        if args.exp == "HICO":
            obj_cat = ho_dict["object_category"]
            obj_name = HICO_OBJECTS[obj_cat]["name"]
            kept_indices = []
            for i, text in enumerate(text_inputs):
                if obj_name in text:
                    kept_indices.append(i)
            kept_indices = torch.tensor(kept_indices).to(device)
            kept_text_features = text_features[kept_indices]
        elif args.exp == "SWIG":
            obj_cat = ho_dict["object_category"]
            obj_name = SWIG_CATEGORIES[obj_cat]["name"]
            kept_indices = []
            for i, text in enumerate(text_inputs):
                if obj_name in text:
                    kept_indices.append(i)
            if len(kept_indices) == 0:
                continue
            kept_indices = torch.tensor(kept_indices).to(device)
            kept_text_features = text_features[kept_indices]
        
        similarity = (100.0 * image_features @ kept_text_features.T).softmax(dim=-1)
        
        if args.exp == "HICO":

            values, indices = similarity[0].topk(min(3, len(similarity[0])))
            preds_per_pair = []
            for score, idx in zip(values, kept_indices[indices]):
                preds_per_pair.append([
                    indices_mapper[int(idx)],
                    ho_dict["person_box"],
                    ho_dict["object_box"],
                    float(score) * ho_dict["person_score"] * ho_dict["object_score"]
                ])
        elif args.exp == "SWIG":
            
            preds_per_pair = []
            for score, idx in zip(similarity[0], kept_indices):
                preds_per_pair.append([
                    indices_mapper[int(idx)],
                    ho_dict["person_box"],
                    ho_dict["object_box"],
                    float(score) * ho_dict["person_score"] * ho_dict["object_score"]
                ])
        predictions.extend(preds_per_pair)

    return predictions
    

def evaluate(args):
    
    if args.exp == "HICO":
        # Load detections
        with open("./baselines/disjoint_detector_clip_dets.pkl", "rb") as f:
            dets = pickle.load(f)
        predictions = {}
        for img_key, dets_per_img in dets.items():
            img_id = int(img_key.split("_")[-1].split(".")[0])
            predictions[img_id] = dets_per_img
        
        # Load and prepare ground truth
        gts = prepare_hico_gts(args.dataset_annos)
    
        hico_ap, hico_rec = hico_evaluation(predictions, gts)
        
        zero_inters = ZERO_SHOT_INTERACTION_IDS
        zero_inters = np.asarray(zero_inters)
        seen_inters = np.setdiff1d(np.arange(600), zero_inters)
        zs_mAP = np.mean(hico_ap[zero_inters])
        sn_mAP = np.mean(hico_ap[seen_inters])
        print("zero-shot mAP: {:.2f}".format(zs_mAP * 100.))
        print("seen mAP: {:.2f}".format(sn_mAP * 100.))
        print("full mAP: {:.2f}".format(np.mean(hico_ap) * 100.))
    
    
        no_inters = NON_INTERACTION_IDS
        zero_inters = np.setdiff1d(zero_inters, no_inters)
        seen_inters = np.setdiff1d(seen_inters, no_inters)
        full_inters = np.setdiff1d(np.arange(600), no_inters)
        zs_mAP = np.mean(hico_ap[zero_inters])
        sn_mAP = np.mean(hico_ap[seen_inters])
        fl_mAP = np.mean(hico_ap[full_inters])
        print("zero-shot mAP: {:.2f}".format(zs_mAP * 100.))
        print("seen mAP: {:.2f}".format(sn_mAP * 100.))
        print("full mAP: {:.2f}".format(fl_mAP * 100.))

    elif args.exp == "SWIG":
        
        # Load and prepare ground truth
        gts, filename_to_id_mapper = prepare_swig_gts(args.dataset_annos)
        
        # Load detections
        with open(f"./outputs/{args.exp}/disjoint_detector_clip_dets.pkl", "rb") as f:
            dets = pickle.load(f)
        predictions = {}
        for img_key, dets_per_img in dets.items():
            img_id = filename_to_id_mapper[img_key]
            predictions[img_id] = dets_per_img

        # Evaluation
        swig_ap, swig_rec = swig_evaluation(predictions, gts)
        
        eval_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1])
        zero_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 0 and x["evaluation"] == 1])
        rare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 1 and x["evaluation"] == 1])
        nonrare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 2 and x["evaluation"] == 1])
        
        full_mAP = np.mean(swig_ap[eval_hois])
        zero_mAP = np.mean(swig_ap[zero_hois])
        rare_mAP = np.mean(swig_ap[rare_hois])
        nonrare_mAP = np.mean(swig_ap[nonrare_hois])
        print("zero-shot mAP: {:.2f}".format(zero_mAP * 100.))
        print("rare mAP: {:.2f}".format(rare_mAP * 100.))
        print("nonrare mAP: {:.2f}".format(nonrare_mAP * 100.))
        print("full mAP: {:.2f}".format(full_mAP * 100.)) 


def main(args):
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    
    # Load dataset
    boxes_dict = load_precomputed_boxes(args)
    
    # Prepare text inputs
    text_features, text_inputs, indices_mapper = prepare_text_inputs(args, model)
    
    predictions = {}
    for img_key, boxes in tqdm(boxes_dict.items()):
        ho_pairs = build_ho_pairs(args, boxes)
        preds = predict(args, model, preprocess, text_features, text_inputs,
                        indices_mapper, img_key, ho_pairs)
        predictions[os.path.basename(img_key)] = preds

    with open(f"./outputs/{args.exp}/disjoint_detector_clip_dets.pkl", "wb") as f:
        pickle.dump(predictions, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    evaluate(args)