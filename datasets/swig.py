# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SWiG-HOI dataset which returns image_id for evaluation.
"""
import os
import json
import torch
import torch.utils.data
from torchvision.datasets import CocoDetection
import datasets.transforms as T
from PIL import Image
from .swig_v1_categories import SWIG_INTERACTIONS, SWIG_ACTIONS, SWIG_CATEGORIES
from utils.sampler import repeat_factors_from_category_frequency, get_dataset_indices

HOI_MAPPER = {(x["action_id"], x["object_id"]): x["id"] for x in SWIG_INTERACTIONS}

class SWiGHOIDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, image_set):
        self.root = img_folder
        self.transforms = transforms
        # Text description of human-object interactions
        dataset_texts, text_mapper = prepare_dataset_text(image_set)
        self.dataset_texts = dataset_texts
        self.text_mapper = text_mapper
        # Load dataset
        repeat_factor_sampling = True if image_set == "train" else False
        reverse_text_mapper = {v: k for k, v in text_mapper.items()}
        self.dataset_dicts = load_swig_json(ann_file, img_folder, reverse_text_mapper, repeat_factor_sampling)

    def __getitem__(self, idx: int):
        
        filename = self.dataset_dicts[idx]["file_name"]
        image = Image.open(filename).convert("RGB")
        
        w, h = image.size
        assert w == self.dataset_dicts[idx]["width"], "image shape is not consistent."
        assert h == self.dataset_dicts[idx]["height"], "image shape is not consistent."

        image_id = self.dataset_dicts[idx]["image_id"]
        annos = self.dataset_dicts[idx]["annotations"]
        
        boxes = torch.as_tensor(annos["boxes"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor(annos["classes"], dtype=torch.int64)
        aux_classes = torch.tensor(annos["aux_classes"], dtype=torch.int64)

        target = {
            "image_id": torch.tensor(image_id),
            "orig_size": torch.tensor([h, w]),
            "boxes": boxes,
            "classes": classes,
            "aux_classes": aux_classes,
            "hois": annos["hois"],
        }
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.dataset_dicts)
    

def load_swig_json(json_file, image_root, text_mapper, repeat_factor_sampling=False):
    """
    Load a json file with HOI's instances annotation.

    Args:
        json_file (str): full path to the json file in HOI instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    imgs_anns = json.load(open(json_file, "r"))

    dataset_dicts = []
    images_without_valid_annotations = []
    for anno_dict in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, anno_dict["file_name"])
        record["height"] = anno_dict["height"]
        record["width"] = anno_dict["width"]
        record["image_id"] = anno_dict["img_id"]
        
        if len(anno_dict["box_annotations"]) == 0 or len(anno_dict["hoi_annotations"]) == 0:
            images_without_valid_annotations.append(anno_dict)
            continue

        boxes = [obj["bbox"] for obj in anno_dict["box_annotations"]]
        classes = [obj["category_id"] for obj in anno_dict["box_annotations"]]
        aux_classes = []
        for obj in anno_dict["box_annotations"]:
            aux_categories = obj["aux_category_id"]
            while len(aux_categories) < 3:
                aux_categories.append(-1)
            aux_classes.append(aux_categories)

        for hoi in anno_dict["hoi_annotations"]:
            target_id = hoi["object_id"]
            object_id = classes[target_id]
            action_id = hoi["action_id"]
            hoi["text"] = generate_text(action_id, object_id)
            continguous_id = HOI_MAPPER[(action_id, object_id)]
            hoi["hoi_id"] = text_mapper[continguous_id]

        targets = {
            "boxes": boxes,
            "classes": classes,
            "aux_classes": aux_classes,
            "hois": anno_dict["hoi_annotations"],
        }

        record["annotations"] = targets
        dataset_dicts.append(record)

    if repeat_factor_sampling:
        repeat_factors = repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh=0.0001)
        dataset_indices = get_dataset_indices(repeat_factors)
        dataset_dicts = [dataset_dicts[i] for i in dataset_indices]
    return dataset_dicts


def generate_text(action_id, object_id):
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

    s = [act, obj_gloss]
    return s

# def generate_text(action_id, object_id):
#     act = SWIG_ACTIONS[action_id]["name"]
#     obj = SWIG_CATEGORIES[object_id]["name"]
#     act_def = SWIG_ACTIONS[action_id]["def"]
#     obj_def = SWIG_CATEGORIES[object_id]["def"]
#     obj_gloss = SWIG_CATEGORIES[object_id]["gloss"]
#     obj_gloss = [obj] + [x for x in obj_gloss if x != obj]
#     if len(obj_gloss) > 1:
#         obj_gloss = " or ".join(obj_gloss)
#     else:
#         obj_gloss = obj_gloss[0]
#     # s = f"A photo of a person {act} with object {obj}. The object {obj} means {obj_def}."
#     # s = f"a photo of a person {act} with object {obj}"
#     # s = f"A photo of a person {act} with {obj}. The {act} means to {act_def}."
#     s = f"A photo of a person {act} with {obj_gloss}. The {act} means to {act_def}."
#     return s


def prepare_dataset_text(image_set):
    texts = []
    text_mapper = {}
    for i, hoi in enumerate(SWIG_INTERACTIONS):
        if image_set != "train" and hoi["evaluation"] == 0: continue
        action_id = hoi["action_id"]
        object_id = hoi["object_id"]
        s = generate_text(action_id, object_id)
        text_mapper[len(texts)] = i
        texts.append(s)
    return texts, text_mapper


def make_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
            T.RandomSelect(
                T.ResizeAndCenterCrop(224),
                T.Compose([
                    T.RandomCrop_InteractionConstraint((0.8, 0.8), 0.9),
                    T.ResizeAndCenterCrop(224)
                ]),
            ),
            normalize
        ])
    if image_set == "val":
        return T.Compose([
            T.ResizeAndCenterCrop(224),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):

    PATHS = {
        "train": ("/raid1/suchen/dataset/swig/images_512",
                  "/raid1/suchen/repo/promting_hoi/data/swig_hoi/swig_trainval_1000.json"),
        "val": ("/raid1/suchen/dataset/swig/images_512",
                "/raid1/suchen/repo/promting_hoi/data/swig_hoi/swig_dev_1000.json"),
        "dev": ("/raid1/suchen/dataset/swig/images_512",
                "/raid1/suchen/repo/promting_hoi/data/swig_hoi/swig_dev_1000.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = SWiGHOIDetection(
        img_folder,
        ann_file,
        transforms=make_transforms(image_set),
        image_set=image_set
    )
    return dataset