from .swig import build as build_swig
from .hico import build as build_hico

from .swig_evaluator import SWiGEvaluator
from .hico_evaluator import HICOEvaluator


def build_dataset(image_set, args):
    if args.dataset_file == 'swig':
        return build_swig(image_set, args)
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


def build_evaluator(args):
    if args.dataset_file == "swig":
        from .swig import SWIG_VAL_ANNO
        evaluator = SWiGEvaluator(SWIG_VAL_ANNO, args.output_dir)
    elif args.dataset_file == "hico":
        from .hico import HICO_VAL_ANNO
        evaluator = HICOEvaluator(HICO_VAL_ANNO, args.output_dir)
    else:
        raise NotImplementedError

    return evaluator