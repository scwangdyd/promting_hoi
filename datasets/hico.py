"""
HICO-DET dataset which returns image_id for evaluation.
"""
import os
import json
import torch
import torch.utils.data
from torchvision.datasets import CocoDetection
import datasets.transforms as T
from PIL import Image
from .hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS, ZERO_SHOT_INTERACTION_IDS
from utils.sampler import repeat_factors_from_category_frequency, get_dataset_indices

id_to_contiguous_id_map = {x["id"]: i for i, x in enumerate(HICO_OBJECTS)}
action_object_to_hoi_id = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

class HICO(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, image_set, zero_shot_exp):
        self.root = img_folder
        self.transforms = transforms
        # Text description of human-object interactions
        dataset_texts, text_mapper = prepare_dataset_text()
        self.dataset_texts = dataset_texts
        self.text_mapper = text_mapper
        # Load dataset
        repeat_factor_sampling = True if image_set == "train" else False
        load_zero_shot = False if zero_shot_exp and image_set == "train" else True
        self.dataset_dicts = load_hico_json(ann_file, img_folder, repeat_factor_sampling, load_zero_shot)

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

        target = {
            "image_id": torch.tensor(image_id),
            "orig_size": torch.tensor([h, w]),
            "boxes": boxes,
            "classes": classes,
            "hois": annos["hois"],
        }
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.dataset_dicts)
    

def load_hico_json(
    json_file,
    image_root,
    repeat_factor_sampling=False,
    load_zero_shot=True
):
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

        if len(anno_dict["annotations"]) == 0 or len(anno_dict["hoi_annotation"]) == 0:
            images_without_valid_annotations.append(anno_dict)
            continue

        boxes = [obj["bbox"] for obj in anno_dict["annotations"]]
        classes = [obj["category_id"] for obj in anno_dict["annotations"]]
        hoi_annotations = []
        for hoi in anno_dict["hoi_annotation"]:
            action_id = hoi["category_id"] - 1 # Starting from 1
            target_id = hoi["object_id"]
            object_id = id_to_contiguous_id_map[classes[target_id]]
            text = (HICO_ACTIONS[action_id]["name"], HICO_OBJECTS[object_id]["name"])
            hoi_id = action_object_to_hoi_id[text]
            
            # Ignore this annotation if we conduct zero-shot simulation experiments
            if (not load_zero_shot) and (hoi_id in ZERO_SHOT_INTERACTION_IDS):
                continue
            
            hoi_annotations.append({
                "subject_id": hoi["subject_id"],
                "object_id": hoi["object_id"],
                "action_id": action_id,
                "hoi_id": hoi_id,
                "text": text
            })

        if (not load_zero_shot) and len(hoi_annotations) == 0:
            continue
        
        targets = {
            "boxes": boxes,
            "classes": classes,
            "hois": hoi_annotations,
        }

        record["annotations"] = targets
        dataset_dicts.append(record)

    if repeat_factor_sampling:
        repeat_factors = repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh=0.003)
        dataset_indices = get_dataset_indices(repeat_factors)
        dataset_dicts = [dataset_dicts[i] for i in dataset_indices]
    return dataset_dicts


def prepare_dataset_text():
    texts = []
    text_mapper = {}
    for i, hoi in enumerate(HICO_INTERACTIONS):
        action_name = " ".join(hoi["action"].split("_"))
        object_name = hoi["object"]
        s = [action_name, object_name]
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
                    T.RandomCrop_InteractionConstraint((0.7, 0.7), 0.9),
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
        "train": ("/raid1/suchen/dataset/hico_20160224_det/images/train2015",
                  "/raid1/suchen/repo/promting_hoi/data/HICO-DET/trainval_hico_ann.json"),
        "val": ("/raid1/suchen/dataset/hico_20160224_det/images/test2015",
                "/raid1/suchen/repo/promting_hoi/data/HICO-DET/test_hico_ann.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = HICO(
        img_folder,
        ann_file,
        transforms=make_transforms(image_set),
        image_set=image_set,
        zero_shot_exp=True,
    )
    return dataset