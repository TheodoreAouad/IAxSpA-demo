import numpy as np
import skimage.morphology as morp


def get_right_left_region(mask, background=0):
    """
    Returns the most left region and the most right region.

    Args:
        mask (ndarray): mask of the regions, pre-labeling. Values are 1 or 0.
        background (int, optional): Value of the background. Defaults to 0.

    Returns:
        ndarray, ndarray: both same shape as mask. The most left region, the most right region.
    """
    regions = morp.label(mask)

    min_centroid = np.infty
    max_centroid = 0
    for label in set(np.unique(regions)).difference(set([background])):
        centroidy = np.where(regions == label)[1].mean()
        if centroidy < min_centroid:
            left = label
            min_centroid = centroidy
        if centroidy > max_centroid:
            right = label
            max_centroid = centroidy

    region_left = np.where(regions == left, mask, background)
    region_right = np.where(regions == right, mask, background)


    return region_left, region_right


def get_iliac_regions(maskil, background=0):
    """
    Get the two regions of the iliac mask. First the left iliac bone, then the right iliac bone.

    Args:
        maskil (ndarray): mask of 1 and 0 of the iliac bone.
        background (int, optional): Value of background. Defaults to 0.

    Returns:
        ndarray, darray: same shapes as maskil. The left iliac, the right iliac.
    """

    mask = get_most_important_regions(maskil, scope=2)
    return get_right_left_region(mask, background)


def get_right_region(mask, background=0):
    return get_right_left_region(mask, background)[1]


def get_left_region(mask, background=0):
    return get_right_left_region(mask, background)[0]


def get_bottom_region(mask, background=0):
    """
    Returns the bottom region.

    Args:
        mask (ndarray): mask of the regions, pre-labeling. Values are 1 or 0.
        background (int, optional): Value of the background. Defaults to 0.

    Returns:
        ndarray: same shape as mask.
    """
    regions = morp.label(mask)

    max_centroid = -1
    for label in set(np.unique(regions)).difference(set([background])):
        centroidy = np.where(regions == label)[0].mean()
        if centroidy > max_centroid:
            bottom = label
            max_centroid = centroidy

    if max_centroid == -1:
        return mask
    return np.where(regions == bottom, mask, background)


def get_all_bbox(region, invert_xy=False, background=0):
    """
    Returns the bounding boxes for each label in the region.

    Args:
        region (ndarray): numpy array with True where we want the bounding box to happen.
        invert_xy (bool): inverts x and y axis.

    Returns:
        tuple: xmin, ymin, xmax+1, ymax+1 with the numpy convention: x first index, y second

    """
    all_bboxes = []
    for val in np.unique(region):
        if val == background:
            continue
        all_bboxes.append(get_bbox(region==val, invert_xy))
    return all_bboxes


def get_bbox(region, invert_xy=False):
    """
    Returns bounding box of the places where region is True.

    Args:
        region (ndarray): numpy array with True where we want the bounding box to happen.
        invert_xy (bool): inverts x and y axis.

    Returns:
        tuple: xmin, ymin, xmax+1, ymax+1 with the numpy convention: x first index, y second
    """
    Xs, Ys = np.where(region)
    xmin, xmax = Xs.min(), Xs.max()
    ymin, ymax = Ys.min(), Ys.max()
    if invert_xy:
        return ymin, xmin, ymax+1, xmax+1
    return xmin, ymin, xmax+1, ymax+1


def get_centroid(region):
    Xs, Ys = np.where(region)
    return np.array([Xs.mean(), Ys.mean()])


def bboxify(img, bbox):
    """
    Given a bounding box with shape xmin, ymin, xmax+1, ymax+1 (numpy convention: x first index, y second index),
    returns the image cropped with this bounding box.

    Args:
        img (ndarray): numpy array to crop
        bbox (tuple-like): bbox with (xmin, ymin, xmax+1, ymax+1)

    Returns:
        ndarray: shape (xmax-xmin, ymax-ymin). Cropped image.
    """
    return img[bbox[0]:bbox[2], bbox[1]:bbox[3]]


def get_side(reg_il):
    cent_il = get_centroid(reg_il)
    y_center = reg_il.shape[1]//2

    return 'left' if cent_il[1] < y_center else 'right'


def check_joint(reg_il, mask_sac, dil_size, side):

    dilated_il = dilation_horizontal(reg_il, side, dil_size)
    return (dilated_il & mask_sac).any()


def dilation_horizontal(reg, side, dil_size):
    xmin, ymin, xmax, ymax = get_bbox(reg)

    if side == 'left':
        from_y = ymin
        to_y = ymax + dil_size
    elif side == 'right':
        from_y = ymin - dil_size
        to_y = ymax

    res = reg + 0
    res[xmin:xmax, from_y:to_y] = 1
    return res


def get_most_important_regions(regions, weights=1, scope=1, background=0):
    """Returns a mask containing only the biggest regions. The mask is labelled.
    The number of regions is the scope.

    Args:
        regions (ndarray): Mask of regions. Either 0 and 1, or already labeled.
        weights (ndarray, optional): weights to give to each label. Defaults to 1.
        scope (int, optional): number of regions. Defaults to 1.
        background (int, optional): pixels that are not part of the most important
                                    regions. Defaults to 0.

    Returns:
        ndarray: same shape as regions. Array like regions but with the less
                important regions being put to background.
    """
    if len(np.unique(regions)) == 2:
        regions = morp.label(regions)
    labels, count = np.unique(regions, return_counts=True)
    count[labels == background] = 0
    weighted = count * weights
    biggest_labels = get_most_important_labels(labels, weighted, scope=scope)
    regions[~np.isin(regions, biggest_labels)] = background
    return regions


def get_most_important_labels(labels, weights, scope=1, return_weights=False):
    """Returns the most weighted elements in labels.

    Args:
        labels (ndarray): array of elements
        weights (ndarray): array of weights for each elements
        scope (int, optional): number of labels to return. Defaults to 1.
        return_weights (bool, optional): If True, returns weights. Defaults to False.

    Returns:
        ndarray: shape (scope,). Labels with biggest weight.
    """
    labels_sorted = labels[weights.argsort()][::-1]
    if return_weights:
        weights_sorted = weights[weights.argsort()][::-1]
        return labels_sorted[:scope], weights_sorted[:scope]
    return labels_sorted[:scope]
