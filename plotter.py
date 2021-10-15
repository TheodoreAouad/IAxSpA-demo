from typing import List

import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import cm, colors


def create_pillow_array_mask(
    rows: List[List[np.ndarray]],
    masks: List[List[np.ndarray]] = [],
    pixels_sep: int = 20,
) -> (np.ndarray, np.ndarray):

    dim0_rows = [max([ar.shape[0] for ar in row if ar is not None]) for row in rows]
    dim0 = sum(dim0_rows)
    dim1 = max([sum([ar.shape[1] for ar in row if ar is not None]) for row in rows])

    sum([max([ar.shape[0] for ar in row if ar is not None]) for row in rows])

    nrows = len(rows)
    ncols = max([len(row) for row in rows])

    big_ar = np.ones((dim0 + pixels_sep*(nrows-1), dim1 + pixels_sep * (ncols - 1)))
    big_mask = np.zeros((dim0 + pixels_sep*(nrows-1), dim1 + pixels_sep * (ncols - 1))) - 1

    prev_i = 0
    for i in range(nrows):
        prev_j = 0
        for j in range(ncols):
            if j < len(rows[i]):

                if rows[i][j] is None:
                    continue

                ar = rows[i][j]
                if i < len(masks) and j < len(masks[i]) and masks[i][j] is not None:
                    big_mask[prev_i:prev_i + ar.shape[0], prev_j:prev_j + ar.shape[1]] = masks[i][j]
                else:
                    big_mask[prev_i:prev_i + ar.shape[0], prev_j:prev_j + ar.shape[1]] = 0

                if ar.max() != ar.min():
                    ar = (ar - ar.min()) / (ar.max() - ar.min())

                big_ar[prev_i:prev_i + ar.shape[0], prev_j:prev_j + ar.shape[1]] = ar
                prev_j = prev_j + ar.shape[1] + pixels_sep
        prev_i = prev_i + dim0_rows[i] + pixels_sep

    return big_ar, big_mask


def create_pillow_image(
    rows: List[List[np.ndarray]],
    masks: List[List[np.ndarray]] = [],
    cmap_img: colors.Colormap = cm.gray,
    cmap_mask: colors.Colormap = cm.jet,
    alpha: float = .5,
    enhance: float = None,
    **kwargs
) -> Image.Image:
    big_ar, big_mask = create_pillow_array_mask(rows, masks, **kwargs)

    pil_ar = array_to_pil(big_ar, mask=big_mask==-1, cmap=cmap_img)
    pil_mask = array_to_pil(big_mask, mask=np.isin(big_mask, [-1, 0]), cmap=cmap_mask)

    pil_mask.putalpha(int(alpha * 255))

    pil_to_save = Image.alpha_composite(pil_ar, pil_mask)

    if enhance:
        enhancer = ImageEnhance.Brightness(pil_to_save)
        pil_to_save = enhancer.enhance(enhance)

    return pil_to_save


def array_to_pil(
    ar: np.ndarray, mask: np.ndarray = None, cmap: colors.Colormap = cm.gray,
) -> Image.Image:
    ar = (ar - ar.min()) / (ar.max() - ar.min())
    if mask is not None:
        ar = np.ma.masked_where(mask, ar)

    return Image.fromarray(np.uint8(cmap(ar)*255))
