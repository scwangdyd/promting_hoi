# promting_hoi

## Baseline 1: Disjoint object detector + CLIP
In the first stage, we use an object detector to generate the bounding boxes for all objects (including humans). Then we pairwisely combine every human and object boxes to build multiple pairs.

In the second stage, for each pair, we crop their union region and send it to the pretrained CLIP model to obtain the interaction prediction.

**Evaluation metrics:** HICO-DET's mAP. A detection is considered as true positive only if (1) both person and object box have an IoU > 0.5 with GT, and (2) the interaction prediction is correct.

To run the baseline 1
```
python baseline_disjoint_detector_and_clip.py \
--exp HICO  \ # or SWIG
--precomputed-boxes [path-to-boxes] \
--dataset-annos [path-to-annotations]
```


## Experimental Results
**HOI Detection on HICO-DET dataset** All experiment settings follow [VCL](https://github.com/zhihou7/HOI-CL).
|              | Unseen |  Seen |  Full |
|--------------|:------:|:-----:|:-----:|
| Shen, WACV18 |  5.62  |   -   |  6.26 |
| FG, AAAI20   |  10.93 | 12.60 | 12.26 |
| VCL, ECCV20  |  10.06 | 24.28 | 21.43 |
| ATL, CVPR21  |  9.18  | 24.67 | 21.57 |
| Baseline 1 (DRG)   |  13.42 | 15.10 | 14.76 |
- Baseline 1 (DRG): using the boxes provided by [DRG](https://github.com/vt-vl-lab/DRG).

**HOI Detection on SWiG-DET dataset**
|              | Unseen | Rare | Non-Rare |
|--------------|:------:|:----:|:--------:|
| PPDM, CVPR20 |  0.78  | 1.62 |   6.53   |
| DIRV, AAAI20 |  0.75  | 1.46 |   5.82   |
| JSR, ECCV20  |  2.34  | 6.10 |   10.01  |
| QMD, ICCV21  |  2.64  | 6.63 |   10.93  |
| Baseline 1 (JSR)  |  2.44  | 4.11 |   7.87   |
* Baseline 1 (JSR): using the boxes computed by [JSR](https://github.com/allenai/swig).
