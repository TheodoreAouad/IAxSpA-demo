import numpy as np
import skimage.morphology as morp
import skimage.measure as meas
import scipy.ndimage as ndimage

from .utils import get_bbox, bboxify, get_bottom_region, \
    get_side, check_joint, dilation_horizontal, get_most_important_regions




def get_roi(
    mask,
    dilation_check=.01,
    inter_dilation_prop=.05,
    closing_size=2,
    iliac_dilation_size=5,
    do_delete_wings=False,
    delete_wings_args={
        'prop_radius': .5,
    },
    *args,
    **kwargs
):
    """
    Get the region of interest (ROI) of the image given the mask of iliac and sacrum.
    The function detects when there is both iliac and sacrum.
    Then, it dilates the iliac and takes the whole dilation. It computes an intersection
    of a dilation of both iliac and sacrum.
    We then perform closing to get the zone between the two bones, and we fill
    the holes.

    Args:
        mask (ndarray): shape (W, L). 0 background, 1 sacrum, 2 iliac.
        inter_dilation_prop (float, optional): Dilation to compute the intersection of iliac and sacrum. Defaults to .1.
        closing_size (int, optional): Size of the closing element. Defaults to 2.
        iliac_dilation_size (int, optional): Size of dilation for iliac bone. Defaults to 5.
        show_time (bool, optional): Shows time to compute every step. Defaults to False.
        return_sacrum (bool, optional): Also gives the ROI corresponding to the sacrum. Defaults to True.

    Returns:
        ndarray: shape (W, L). 0 background, 1 for ROI.
    """


    mask_il = mask == 2
    mask_sac = mask == 1

    if mask_il.sum() == 0:
        return np.zeros((1, 1))

    res = np.zeros_like(mask).astype(bool)

    labels_il = morp.label(mask_il)

    for label in set(np.unique(labels_il)).difference([0]):
        reg_il = labels_il == label
        side = get_side(reg_il)

        dil_size_check = int(dilation_check * reg_il.shape[1])

        is_joint = check_joint(reg_il, mask_sac, dil_size_check, side)
        if not is_joint:
            continue

        if do_delete_wings:
            reg_il = crop_iliac(reg_il, do_bboxify=False, **delete_wings_args)

            if reg_il.sum() == 0:
                continue


        selem_iliac = morp.disk(iliac_dilation_size)
        reg_il = morp.dilation(reg_il, selem_iliac)

        dil_size = int(inter_dilation_prop * reg_il.shape[1])
        extended_il = dilation_horizontal(reg_il, side, dil_size)
        intersec = extended_il & mask_sac


        res = res | (reg_il.astype(bool) | intersec)

    res = morp.closing(res, morp.disk(closing_size))
    res = ndimage.binary_fill_holes(res)

    return res


def crop_roi(roi_mask, ):
    """
    This functions takes the whole roi mask and crops the ROIs.

    Args:
        roi_mask (ndarary): shape (W, L). 1 and 0.

    Returns:
        ndarray, ndarray: Shapes smaller than the input. Only contains the ROIs.
                          Left then right.
    """

    if not roi_mask.any():
        return np.zeros_like(roi_mask), np.zeros_like(roi_mask)


    mid = roi_mask.shape[1]//2
    rois = [roi_mask + 0 for _ in range(2)]

    rois[0][:, mid:] = 0
    rois[1][:, :mid] = 0

    for idx in range(len(rois)):
        rois[idx] = get_most_important_regions(rois[idx], scope=1) > 0

    return rois


def crop_roi_old(roi_mask,):
    """
    This functions takes the whole roi mask and crops the ROIs.

    Args:
        roi_mask (ndarary): shape (W, L). 1 and 0.

    Returns:
        ndarray, ndarray: Shapes smaller than the input. Only contains the ROIs.
                          Left then right.
    """

    if not roi_mask.any():
        return np.zeros_like(roi_mask), np.zeros_like(roi_mask)


    labels = morp.label(roi_mask)
    if len(np.unique(labels)) == 2:
        roi_mask_split = roi_mask + 0
        roi_mask_split[:, roi_mask.shape[1]//2] = 0
        labels = morp.label(roi_mask_split)
    rois_important = get_most_important_regions(labels, scope=2)
    rois = [np.zeros_like(roi_mask) for _ in range(2)]

    min_centroid = np.infty
    left = 0
    for idx, roim in enumerate(meas.regionprops(rois_important)):
        xmin, ymin, xmax, ymax = roim.bbox
        rois[idx][xmin:xmax, ymin:ymax] = roim.image
        if roim.centroid[1] < min_centroid:
            left = idx
            min_centroid = roim.centroid[1]

    return rois[left], rois[1 - left]


def crop_iliac(il_mask, prop_radius=.5, do_bboxify=True, selem_type='circle'):
    """
    Given the mask of a joint, deletes the wing of the iliac using an opening.

    Args:
        il_mask (ndarray): Arrays of 0 and 1 indicating the mask of the joint
        prop_radius (float, optional): Proportion of the radius for the selem opening. Defaults to .5.
        do_bboxify (bool, optional): Returns the bbox of the region. If False, returns a mask
                                    with the same size as il_mask. Defaults to True.

    Returns:
        ndarray, (tuple): If do_bboxify, returns a mask with shape the bbox of the joint, and the bbox.
                            If not do_bboxify, returns a mask with same size as il_mask.
    """

    bbox = get_bbox(il_mask)
    crop = bboxify(il_mask, bbox)

    dist = ndimage.morphology.distance_transform_edt(crop)
    skel = morp.skeletonize(crop)

    maxr = int(max(dist[skel]) * prop_radius)
    if selem_type == 'circle':
        selem = morp.disk(maxr)
    elif selem_type == 'square':
        selem = morp.square(maxr*2)

    opened = morp.opening(crop, selem)

    xmin = np.where(
        get_bottom_region(opened)
    )[0][0]
    if do_bboxify:
        return np.where(np.arange(len(crop))[:, None] > xmin, crop, 0), bbox
    return np.where(np.arange(len(il_mask))[:, None] > xmin + bbox[0], il_mask, 0)



def delete_wings_iliac(il_target, **crop_iliac_args):
    """
    Given an iliac mask, deletes the wing for  both iliac. Assumes
    2 regions only for the iliac mask.

    Args:
        il_target (ndarray): numpy array of 1 where there is an iliac, 0 elsewhere.
                             Assumed to be only 2 regions of iliac.
        **crop_iliac_args:
            prop_radius (float, optional): Proportion of the radius for the selem opening. Defaults to .5.
            do_bboxify (bool, optional): Returns the bbox of the region. If False, returns a mask
                                        with the same size as il_mask. Defaults to True.
    Returns:
        ndarray: same shape as il_target where the wings are deleted.
    """
    res = il_target + 0
    labls = morp.label(res)
    for i in set(np.unique(labls)).difference([0]):

        reg, bbox = crop_iliac(labls == i, **crop_iliac_args)
        res[bbox[0]:bbox[2], bbox[1]:bbox[3]] = reg
    return res


def delete_wings(mask, **delete_wings_args):
    """
    Given a sacro-iliac mask, deletes the wings
    for the iliac.

    Args:
        mask (ndarray): numpy array containing the sacro-iliac mask. (sacrum: 1, iliac: 2, background: 0)

    Returns:
        ndarray: same shape as mask, where the iliac have had their wings deleted.
    """
    ilt = delete_wings_iliac(mask==2, **delete_wings_args)
    return np.where(ilt, 2, mask==1)
