from typing import Tuple

import torch
import pandas as pd


def process_single_output(output: "detectron2.structures.instances.Instances" = None,) -> pd.DataFrame:
    """ Converts an output of the model (and the ground truth) in a nice
    dataframe with the important information.

    Args:
        dataset_dict (dict): Ground Truth. Format described in
            https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
        output (detectron2.structures.instances.Instances): output of the model
            to be processed/

    Returns:
        pandas.core.frame.DataFrame: dataframe of len 1 with the important info
    """

    res = dict(
        **{f'nb_boxes_pred_{side}': 0 for side in ['right', 'left']},
        **{f'{type_pred}_pred_{side}': [] for type_pred in ['boxes', 'mask', 'label'] for side in ['right', 'left']},
        **{f'pred_scores_{side}': [] for side in ['right', 'left']},
        **{f'label_true_{side}': -1 for side in ['right', 'left']},
    )

    for idx in range(len(output['instances'])):
        pred = output['instances'][idx]
        side = get_side_annotation(pred, pred=True)
        res['boxes_pred' + '_' + side] = res['boxes_pred' + '_' + side] + [pred.get('pred_boxes').tensor.to("cpu").numpy()]
        res['mask_pred' + '_' + side] = res['mask_pred' + '_' + side] + [pred.get('pred_masks').to("cpu")]
        res['label_pred' + '_' + side] = res['label_pred' + '_' + side] + [pred.get('pred_classes').to("cpu").item()]
        res['pred_scores' + '_' + side] = res['pred_scores' + '_' + side] + [pred.get('scores').to("cpu").item()]
        res['nb_boxes_pred' + '_' + side] += 1

    res.update({k: v for k, v in output.items() if k != 'instances'})

    for side in ['right', 'left']:
        for col in ['label_pred_', 'pred_scores_']:
            if len(res[col + side]) == 0:
                res[col + side] = [-1]

    for side in ['right', 'left']:
        for col in ['mask_pred_', 'boxes_pred_']:
            if len(res[col + side]) == 0:
                res[col + side] = [torch.zeros([1] + list(output['instances']._image_size)).bool()]

    res = pd.DataFrame({k: [v] for k, v in res.items()})

    return res


def get_side_boxes(shape: Tuple, bbox: Tuple, invert_bbox: bool = False) -> str:
    """ Given a bounding box and the shape of the corresponding image,
    returns the side of the bounding box (either on the left side or right side
    of the image).

    Args:
        shape (array-like): shape of the image the box is inside
        bbox (array-like): (xmin, ymin, xmax, ymax)
        invert_bbox (bool): inverts x and y coordinate

    Returns:
        str: either 'left' or 'right'
    """
    xmin, ymin, xmax, ymax = bbox
    if invert_bbox:
        ymin, xmin, ymax, xmax = bbox

    ymed = shape[1] // 2
    if (ymax + ymin) // 2 < ymed:
        return 'left'
    return 'right'


def get_side_annotation(annot: "detectron2.structures.instances.Instances", pred: bool = True) -> str:
    """ Given the annotation in the format of the dataset_dict
    (see https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html),
    returns the side of the annotation.

    Args:
        annot (detectron2.structures.instances.Instances): output of the model
        pred (bool): whether the annotation type is a prediction of the model
            or the  training type.

    Returns:
        str: either 'left' or 'right'
    """
    if pred:
        box = annot.get('pred_boxes').tensor.to("cpu").numpy()[0]
        return get_side_boxes(annot.image_size, box, True)
    return get_side_boxes(annot['segmentation']['size'], annot['bbox'], True)
