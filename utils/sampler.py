import math
from collections import defaultdict
import torch


def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors based on category frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.

        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        # 1. For each interaction c, compute the fraction of images that contain it: f(c)
        interaction_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cats = set()
            for hoi in dataset_dict["annotations"]["hois"]:
                cats.add(hoi["hoi_id"])
            for cat_id in cats:
                interaction_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in interaction_freq.items():
            interaction_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in interaction_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for i, dataset_dict in enumerate(dataset_dicts):
            cats = set()
            for hoi in dataset_dict["annotations"]["hois"]:
                cats.add(hoi["hoi_id"])
            rep_factor = max({category_rep[cat_id] for cat_id in cats})
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)


def get_dataset_indices(repeat_factors):
    g = torch.Generator()
    # Split into whole number (_int_part) and fractional (_frac_part) parts.
    _int_part = torch.trunc(repeat_factors)
    _frac_part = repeat_factors - _int_part
    
    # Since repeat factors are fractional, we use stochastic rounding so
    # that the target repeat factor is achieved in expectation over the
    # course of training
    rands = torch.rand(len(_frac_part), generator=g)
    rep_factors = _int_part + (rands < _frac_part).float()
    # Construct a list of indices in which we repeat images as specified
    indices = []
    for dataset_index, rep_factor in enumerate(rep_factors):
        indices.extend([dataset_index] * int(rep_factor.item()))
    return indices