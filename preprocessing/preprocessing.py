from skimage.exposure import equalize_adapthist
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import disk
from skimage.filters import median

from .processing import ProcessImage


class TorchMinMaxNorm():

    def __call__(self, tensor):
        return self.min_max_norm(tensor)

    @staticmethod
    def min_max_norm(tensor):
        mins = tensor.min()
        maxs = tensor.max()
        return tensor.sub_(mins).div_(maxs - mins)


class ToUint8(ProcessImage):

    def apply_to_img2d(self, img_orig):
        img = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())
        return (img * 255).astype(np.uint8)


class ConstantPixelSpacing(ProcessImage):

    def __init__(
            self,
            pixel_spacing,
            # background=None,
            interpolation_input=cv2.INTER_LINEAR,
            interpolation_target=cv2.INTER_NEAREST,
            apply_to_target=False,
            channels=None
    ):
        super().__init__(channels)
        if type(pixel_spacing) in [float, int, np.int64]:
            self.pixel_spacing = (pixel_spacing, pixel_spacing)
        else:
            self.pixel_spacing = pixel_spacing
        assert len(self.pixel_spacing) == 2, "pixel_spacing must be len 2, {}".format(pixel_spacing)
        self.interpolation_input = interpolation_input
        self.interpolation_target = interpolation_target
        self.apply_to_target = apply_to_target
        # self.bg = background

    def apply_to_img2d(self, img, pixel_spacing, interpolation, *args, **kwargs):
        rf = np.zeros(2).astype(int)
        for i in range(len(img.shape)):
            rf[i] = int(img.shape[i] * pixel_spacing[i] / self.pixel_spacing[i])
        return cv2.resize(img, tuple(rf), interpolation=interpolation,)
        # return resize_bg(img, tuple(rf)[::-1], background=self.bg, interpolation=interpolation)


    def apply_to_row(self, row_original):
        row = row_original.copy()

        if type(self.interpolation_input) == int:
            if len(row.pixel_array.shape) == 3:
                interpolation_input = [[self.interpolation_input] for _ in range(row.pixel_array.shape[-1])]
            elif len(row.pixel_array.shape) == 2:
                interpolation_input = [self.interpolation_input]
        else:
            interpolation_input = self.interpolation_input

        row.pixel_array = self.apply_to_img(
            row.pixel_array,
            row.PixelSpacing,
            args_per_chan=interpolation_input,
        )


        # if 'target' in row.keys() and row['target'].dtype == np.ndarray and len(row['target'].shape) > 1:
        if self.apply_to_target and 'target' in row.keys() and row['target'] is not None:
            if type(self.interpolation_target) == int:
                if len(row.pixel_array.shape) == 3:
                    interpolation_target = [[self.interpolation_target] for _ in range(row.pixel_array.shape[-1])]
                elif len(row.pixel_array.shape) == 2:
                    interpolation_target = [self.interpolation_target]
            else:
                interpolation_target = self.interpolation_target

            row.target = self.apply_to_img(
                row.target,
                row.PixelSpacing,
                args_per_chan=interpolation_target,
            )

            assert row.target.shape == row.pixel_array.shape, "Target and Image not same shape"

        row['OldPixelSpacing'] = row['PixelSpacing']
        row['PixelSpacing'] = self.pixel_spacing
        row['resolution'] = (row.pixel_array.shape[0], row.pixel_array.shape[1])

        return row


class MinMaxNorm(ProcessImage):

    def __init__(self, background=None, channels=None):
        super().__init__(channels)
        self.bg = background

    def apply_to_img2d(self, img, *args, **kwargs):
        if self.bg is None:
            return (img - img.min()) / (img.max() - img.min())
        res = (img + 0).astype(float)
        res[img!=self.bg] = (img[img!=self.bg] - img[img!=self.bg].min()) / (img[img!=self.bg].max() - img[img!=self.bg].min())
        return res



class RemoveGaussianBias(ProcessImage):

    def __init__(self, coef_smoothing=.05, non_zero_divide=1e-6, channels=None):
        super().__init__(channels)
        self.coef_smoothing = coef_smoothing
        self.non_zero_divide = non_zero_divide

    def apply_to_img2d(self, img, *args, **kwargs):

        sigma = (img.shape[0] * self.coef_smoothing, img.shape[1] * self.coef_smoothing)
        cor = ndimage.gaussian_filter(img, sigma=sigma)
        return img /(cor + self.non_zero_divide)


class LocalMedian(ProcessImage):

    def __init__(self, region=disk(1), background=None, channels=None):
        super().__init__(channels)
        self.region = region
        self.bg = background


    def apply_to_img2d(self, img, *args, **kwargs):
        if self.bg is not None:
            # TODO: add again the mask
            # med = median(img, self.region, mask=(img != self.bg))
            med = median(img, self.region)
        else:
            med = median(img, self.region)
        res = med / med.max()      # Median is in [0, 255]
        if self.bg is not None:
            res[img == self.bg] = self.bg
        return res


class EqualizeAdaptHist(ProcessImage):

    def apply_to_img2d(self, img, *args, **kwargs):
        return equalize_adapthist(img)
