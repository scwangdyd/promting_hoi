"""
HICO-DET dataset utils
"""
import os
import json
import collections
import torch
import torch.utils.data
from torchvision.datasets import CocoDetection
import datasets.transforms as T
from PIL import Image
from .hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS, ZERO_SHOT_INTERACTION_IDS, NON_INTERACTION_IDS
from utils.sampler import repeat_factors_from_category_frequency, get_dataset_indices


# NOTE: Replace the path to your file
HICO_TRAIN_ROOT = "./data/hico_20160224_det/images/train2015"
HICO_TRAIN_ANNO = "./data/hico_20160224_det/annotations/trainval_hico_ann.json"
HICO_VAL_ROOT = "./data/hico_20160224_det/images/test2015"
HICO_VAL_ANNO = "./data/hico_20160224_det/annotations/test_hico_ann.json"


class HICO(CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        image_set,
        zero_shot_exp,
        repeat_factor_sampling,
        ignore_non_interaction
    ):
        """
        Args:
            json_file (str): full path to the json file in HOI instances annotation format.
            image_root (str or path-like): the directory where the images in this json file exists.
            transforms (class): composition of image transforms.
            image_set (str): 'train', 'val', or 'test'.
            repeat_factor_sampling (bool): resampling training data to increase the rate of tail
                categories to be observed by oversampling the images that contain them.
            zero_shot_exp (bool): if true, see the last 120 rare HOI categories as zero-shot,
                excluding them from the training data. For the selected rare HOI categories, please
                refer to `<datasets/hico_categories.py>: ZERO_SHOT_INTERACTION_IDS`.
            ignore_non_interaction (bool): Ignore non-interaction categories, since they tend to
                confuse the models with the meaning of true interactions.
        """
        self.root = img_folder
        self.transforms = transforms
        # Text description of human-object interactions
        dataset_texts, text_mapper = prepare_dataset_text()
        self.dataset_texts = dataset_texts
        self.text_mapper = text_mapper # text to contiguous ids for evaluation
        object_to_related_hois, action_to_related_hois = prepare_related_hois()
        self.object_to_related_hois = object_to_related_hois
        self.action_to_related_hois = action_to_related_hois
        # Load dataset
        repeat_factor_sampling = repeat_factor_sampling and image_set == "train"
        zero_shot_exp = zero_shot_exp and image_set == "train"
        self.dataset_dicts = load_hico_json(
            json_file=ann_file,
            image_root=img_folder,
            zero_shot_exp=zero_shot_exp,
            repeat_factor_sampling=repeat_factor_sampling,
            ignore_non_interaction=ignore_non_interaction)

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
    zero_shot_exp=True,
    repeat_factor_sampling=False,
    ignore_non_interaction=True,
):
    """
    Load a json file with HOI's instances annotation.

    Args:
        json_file (str): full path to the json file in HOI instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        repeat_factor_sampling (bool): resampling training data to increase the rate of tail
            categories to be observed by oversampling the images that contain them.
        zero_shot_exp (bool): if true, see the last 120 rare HOI categories as zero-shot,
            excluding them from the training data. For the selected rare HOI categories, please
            refer to `<datasets/hico_categories.py>: ZERO_SHOT_INTERACTION_IDS`.
        ignore_non_interaction (bool): Ignore non-interaction categories, since they tend to
            confuse the models with the meaning of true interactions.
    Returns:
        list[dict]: a list of dicts in the following format.
        {
            'file_name': path-like str to load image,
            'height': 480,
            'width': 640,
            'image_id': 222,
            'annotations': {
                'boxes': list[list[int]], # n x 4, bounding box annotations
                'classes': list[int], # n, object category annotation of the bounding boxes
                'hois': [
                    {
                        'subject_id': 0,  # person box id (corresponding to the list of boxes above)
                        'object_id': 1,   # object box id (corresponding to the list of boxes above)
                        'action_id', 76,  # person action category
                        'hoi_id', 459,    # interaction category
                        'text': ('ride', 'skateboard') # text description of human action and object
                    }
                ]
            }
        }
    """
    imgs_anns = json.load(open(json_file, "r"))

    id_to_contiguous_id_map = {x["id"]: i for i, x in enumerate(HICO_OBJECTS)}
    action_object_to_hoi_id = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

    dataset_dicts = []
    images_without_valid_annotations = []
    for anno_dict in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, anno_dict["file_name"])
        record["height"] = anno_dict["height"]
        record["width"] = anno_dict["width"]
        record["image_id"] = anno_dict["img_id"]

        ignore_flag = False
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
            if zero_shot_exp and (hoi_id in ZERO_SHOT_INTERACTION_IDS):
                ignore_flag = True
                continue

            # Ignore non-interactions
            if ignore_non_interaction and action_id == 57:
                continue

            hoi_annotations.append({
                "subject_id": hoi["subject_id"],
                "object_id": hoi["object_id"],
                "action_id": action_id,
                "hoi_id": hoi_id,
                "text": text
            })

        if len(hoi_annotations) == 0 or ignore_flag:
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


def prepare_related_hois():
    ''' Gather related hois based on object names and action names
    Returns:
        object_to_related_hois (dict): {
            object_text (e.g., chair): [
                {'hoi_id': 86, 'text': ['carry', 'chair']},
                {'hoi_id': 87, 'text': ['hold', 'chair']},
                ...
            ]
        }

        action_to_relatedhois (dict): {
            action_text (e.g., carry): [
                {'hoi_id': 10, 'text': ['carry', 'bicycle']},
                {'hoi_id': 46, 'text': ['carry', 'bottle']},
                ...
            ]
        }
    '''
    object_to_related_hois = collections.defaultdict(list)
    action_to_related_hois = collections.defaultdict(list)

    for x in HICO_INTERACTIONS:
        action_text = x['action']
        object_text = x['object']
        hoi_id = x['interaction_id']
        if hoi_id in ZERO_SHOT_INTERACTION_IDS or hoi_id in NON_INTERACTION_IDS:
            continue
        hoi_text = [action_text, object_text]

        object_to_related_hois[object_text].append({'hoi_id': hoi_id, 'text': hoi_text})
        action_to_related_hois[action_text].append({'hoi_id': hoi_id, 'text': hoi_text})

    return object_to_related_hois, action_to_related_hois


def make_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])

    scales = [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
            T.RandomSelect(
                T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                T.Compose([
                    T.RandomCrop_InteractionConstraint((0.75, 0.75), 0.8),
                    T.RandomResize(scales, max_size=scales[-1] * 1333 // 800),
                ])
            ),
            normalize,
        ])

    if image_set == "val":
        return T.Compose([
            T.RandomResize([args.eval_size], max_size=args.eval_size * 1333 // 800),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')

    """ deprecated (Fixed image resolution + random cropping + centering)
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
    """


def build(image_set, args):
    # NOTE: Replace the path to your file
    PATHS = {
        "train": (HICO_TRAIN_ROOT, HICO_TRAIN_ANNO),
        "val": (HICO_VAL_ROOT, HICO_VAL_ANNO),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = HICO(
        img_folder,
        ann_file,
        transforms=make_transforms(image_set, args),
        image_set=image_set,
        zero_shot_exp=args.zero_shot_exp,
        repeat_factor_sampling=args.repeat_factor_sampling,
        ignore_non_interaction=args.ignore_non_interaction
    )

    return dataset