import os
from os.path import join
import warnings

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pathlib
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from utils import ceil_, log_console
from load_mri import get_mri
from roi.cropper import get_roi, crop_roi
from roi.utils import get_bbox, bboxify
from plotter import create_pillow_image
from process_output_detectron2 import process_single_output


class Patient:
    """
    Base class for patient. Contains all relevent information about a patient.
    Can be used to compute segmentation, ROI, classification in one object.
    """


    def __init__(self, patient_id=None, do_warn=True, do_exception=False, *args, **kwargs):
        self.patient_id = patient_id

        self.mri_t1, self.mri_stir = self._get_mris(*args, **kwargs)
        self.check_mris(do_warn=do_warn, do_exception=do_exception)

        self.is_positive = None
        self.segmentation = None

        self.roi_mask = None
        self.crop_left = None
        self.crop_right = None

        self.classif_data = None

        self.logger = None


    def check_mris(self, do_warn=True, do_exception=False):
        assert (self.mri_stir is not None) and (self.mri_t1 is not None), "self.get_mri must not return None"

        # Check Length
        if len(self.mri_stir) != len(self.mri_t1):
            msg = (""
                "The STIR and T1 do not have the same number of slices. "
                f"T1: {len(self.mri_t1)}, STIR: {len(self.mri_stir)}. This could lead to a problem in the future."
            "")
            if do_exception:
                raise ValueError(msg)
            if do_warn:
                warnings.warn(msg)

        # Check Cosdirs
        if self.mri_stir.cosdirs.iloc[0] != self.mri_t1.cosdirs.iloc[0]:
            msg = (""
                "The STIR and T1 slices do not have the same orientation."
            "")
            if do_exception:
                raise ValueError(msg)
            if do_warn:
                warnings.warn(msg)

        # Check other DICOM values
        if len(self.mri_stir) == len(self.mri_t1):
            for col in ['ImagePosition', "SliceThickness"]:
                if (self.mri_t1[col] != self.mri_stir[col]).any():
                    msg = (""
                        f"The STIR and T1 do not have the same {col}."
                    "")
                    if do_exception:
                        raise ValueError(msg)
                    if do_warn:
                        warnings.warn(msg)


    def set_logger(self, logger):
        self.logger = logger


    def remove_logger(self):
        self.logger = None


    def log_console(self, to_print, *args, level='info', **kwargs):
        log_console(to_print, *args, level=level, logger=self.logger, **kwargs)


    def _get_mris(self, *args, **kwargs):
        """
        Child class should implement
        """
        raise NotImplementedError


    def _get_volume(self, mri, axis=-1):
        return Patient.get_volume_from_mri(mri, axis_to_stack=axis)


    @property
    def depth(self):
        return len(self.mri_t1)


    @property
    def pixel_spacing(self):
        return self.mri_t1.PixelSpacing.iloc[0]


    @property
    def cosdirs(self):
        return self.mri_t1.cosdirs.iloc[0]


    @property
    def cube_t1(self):
        return self._get_volume(self.mri_t1)


    @property
    def cube_stir(self):
        return self._get_volume(self.mri_stir)


    def _dl_segment_volume(
        self,
        preprocessing,
        model,
        batch_size,
        batch_preprocessing,
        device='cpu',
    ):
        """ Performs segmentation using deep learning.

        Args:
            preprocessing (function): takes as input (W, L, D), outputs (W', L', D')
            model (nn.Module child): deep learning model
            batch_size (int): number of images to segment at the same time in model
            batch_preprocessing (function): processing to give to the dataloader
            device (str, optional): GPU or CPU. Defaults to 'cpu'.
        """
        all_inpt = preprocessing(self.cube_t1)

        n_batch = ceil_(self.depth / batch_size)

        all_outputs = []
        with torch.no_grad():
            model.to(device)
            for batch_idx in range(n_batch):
                inpt = []
                for i in range(batch_idx * batch_size, min(self.depth, (batch_idx + 1) * batch_size)):
                    inpt.append(batch_preprocessing(all_inpt[..., i]))
                inpt = torch.stack(inpt).to(device)
                all_outputs.append(model(inpt).cpu())

        all_outputs = torch.cat(all_outputs).detach().numpy()
        self.segmentation = all_outputs.argmax(1).transpose(1, 2, 0).astype(np.int16)
        return self.segmentation


    def _load_segment_volume(
        self,
        path='',
    ):
        self.segmentation = np.load(path).astype(np.int16)
        return self.segmentation


    def segment_volume(
        self,
        segm_type='dl',
        segm_args={},
    ):
        """
        Performs segmentation.

        Args:
            segm_type (str, optional): Which type of segmentation to perform. Defaults to 'dl'.
            segm_args (dict, optional): Args to perform the segmentation. Defaults to {}.

        Returns:
            [type]: [description]
        """
        if segm_type == 'dl':
            return self._dl_segment_volume(**segm_args)

        if segm_type == 'load':
            return self._load_segment_volume(**segm_args)



    @property
    def sacrum(self):
        return self.segmentation == 1


    @property
    def iliac(self):
        return self.segmentation == 2


    def get_rois(
        self,
        do_delete_wings=True,
        delete_wings_args={},
        cropper_args={},
    ):
        """
        Computes Regions of Interest (ROIs).

        Args:
            do_delete_wings (bool, optional): Deletes wings of iliac. Defaults to True.
            delete_wings_args (dict, optional): Args for deleting wings. Defaults to {}.
            cropper_args (dict, optional): Args to cropping. Defaults to {}.
        """
        assert self.segmentation is not None, 'Segment First.'
        all_slices = range(self.segmentation.shape[-1])
        all_roi_mask = []
        all_crops_left = []
        all_crops_right = []
        for slice_idx in all_slices:

            roi_mask = get_roi(
                self.segmentation[..., slice_idx],
                do_delete_wings=do_delete_wings,
                delete_wings_args=delete_wings_args,
                **cropper_args,
            )
            all_roi_mask.append(roi_mask[..., None])

            crop_left, crop_right = crop_roi(roi_mask)
            all_crops_left.append(crop_left[..., None])
            all_crops_right.append(crop_right[..., None])

        self.roi_mask = np.concatenate(all_roi_mask, axis=-1)
        self.crop_left = np.concatenate(all_crops_left, axis=-1)
        self.crop_right = np.concatenate(all_crops_right, axis=-1)


    def pred_asas_positive(self, thresh=2):
        assert 'preds' in self.classif_data.columns
        return self.preds.sum() >= thresh

    def save_slices(self, savepath, verbose=1):
        pathlib.Path(savepath).mkdir(exist_ok=True, parents=True)

        iterator = range(self.depth)
        if verbose:
            iterator = tqdm(iterator)
            self.log_console('Saving slices on {} ...'.format(savepath))

        all_images = []

        for i in iterator:
            rows = [[None, None]]
            for i1, sequence in enumerate([self.cube_t1[..., i], self.cube_stir[..., i]]):
                rows[0][i1] = sequence

            pil_to_save = create_pillow_image(rows=rows, enhance=3)
            pil_to_save.save(join(savepath, 'slice_{}.png'.format(i)))
            all_images.append(pil_to_save)

        if verbose:
            self.log_console('Saved slices on {}.'.format(savepath))

        return all_images

    def save_rois(self, savepath, verbose=1):
        assert self.crop_left is not None, 'Must compute ROIs first.'
        pathlib.Path(join(savepath, 'png')).mkdir(exist_ok=True, parents=True)

        iterator = range(self.depth)
        if verbose:
            iterator = tqdm(iterator)
            self.log_console('Saving ROIs on {} ...'.format(savepath))

        np.save(join(savepath, 'right_roi.npy'), self.crop_right)
        np.save(join(savepath, 'left_roi.npy'), self.crop_left)

        all_images = []

        for i in iterator:
            full_none = True
            rows = [[None, None, None, None], [None, None, None, None]]
            for i1, sequence in enumerate([self.cube_t1[..., i], self.cube_stir[..., i]]):
                for j1, crop_side in enumerate([self.crop_left[..., i], self.crop_right[..., i]]):
                    if crop_side.sum() == 0:
                        continue
                    rows[i1][j1] = bboxify(sequence, get_bbox(crop_side))
                    rows[i1][j1+2] = bboxify(sequence*crop_side, get_bbox(crop_side))
                    full_none = False

            if full_none:
                rows = [[np.zeros((10, 10))]]
            pil_to_save = create_pillow_image(rows=rows, enhance=3)
            pil_to_save.save(join(savepath, 'png', 'roi_{}.png'.format(i)))
            all_images.append(pil_to_save)

        if verbose:
            self.log_console('Saved ROIs on {}.'.format(savepath))

        return all_images

    def save_segmentation(self, savepath, verbose=1):
        assert self.segmentation is not None, 'Segmentation not computed.'

        pathlib.Path(join(savepath, 'png')).mkdir(exist_ok=True, parents=True)

        iterator = range(self.depth)
        if verbose:
            iterator = tqdm(iterator)
            self.log_console('Saving segmentation on {} ...'.format(savepath))

        np.save(join(savepath, 'segmentation.npy'), self.segmentation)

        all_images = []

        for i in iterator:
            pil_to_save = create_pillow_image(
                rows=[
                    [self.cube_t1[..., i], self.cube_t1[..., i]]
                ],
                masks=[
                    [None, self.segmentation[..., i]]
                ],
                enhance=3,
            )

            pil_to_save.save(join(savepath, 'png', 'segm_{}.png'.format(i)))
            all_images.append(pil_to_save)

        if verbose:
            self.log_console('Saved segmentation on {}.'.format(savepath))

        return all_images


    def apply_detectron2(self, predictor, preprocessing=None, t1_chans=None, stir_chans=None, save_inputs_path=None):

        if t1_chans is None:
            t1_chans = self.t1_chans
        if stir_chans is None:
            stir_chans = self.stir_chans

        all_outputs = []
        all_inputs = []
        # for t1_slice, stir_slice in zip(self.mri_t1['pixel_array'], self.mri_stir['pixel_array']):
        for idx in tqdm(range(len(self.mri_t1))):
            t1_slice = self.mri_t1['pixel_array'].iloc[idx]
            stir_slice = self.mri_stir['pixel_array'].iloc[idx]
            pixel_array = np.zeros(list(t1_slice.shape) + [max(max(t1_chans), max(stir_chans)) + 1])

            for chan in t1_chans:
                pixel_array[..., chan] = t1_slice
            for chan in stir_chans:
                pixel_array[..., chan] = stir_slice

            if preprocessing is not None:
                temp_df = pd.DataFrame({'pixel_array': [pixel_array], 'PixelSpacing': [self.mri_t1['PixelSpacing'].iloc[0]]})
                pixel_array = preprocessing.train(temp_df)['pixel_array'].iloc[0]

            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
            pixel_array = (pixel_array * 255).astype(np.uint8)
            cur_output = predictor(pixel_array)
            cur_output.update({
                'slice_nb': self.mri_t1['slice_nb'].iloc[idx] if 'slice_nb' in self.mri_t1.columns else idx,
                'path_slice': self.mri_t1['path_slice'].iloc[idx] if 'path_slice' in self.mri_t1.columns else 'path',
            })

            all_outputs.append(cur_output)
            all_inputs.append(pixel_array)

        self.all_inputs = all_inputs
        if save_inputs_path is not None:
            self.save_inputs(save_inputs_path)

        self.detectron2_outputs = all_outputs
        return self.detectron2_outputs


    def save_inputs(self, save_inputs_path):
        pathlib.Path(save_inputs_path).mkdir(exist_ok=True, parents=True)
        for idx in range(len(self.all_inputs)):
            pixel_array = self.all_inputs[idx]
            pil_to_save = create_pillow_image(rows=[
                [pixel_array[..., 0], pixel_array[..., 1], pixel_array[..., 2]]
            ])
            pil_to_save.save(join(save_inputs_path, f'input_{idx}.png'))


    def create_df_detectron2_outputs(self):
        all_dfs = []
        for output in self.detectron2_outputs:
            cur_df = process_single_output(output=output)
            cur_df['patient_id'] = self.patient_id
            all_dfs.append(cur_df)
        df_results = pd.concat(all_dfs)
        patient_diagnosis = self.decision_rule_threshold(df_results)
        df_results['patient_diagnosis'] = patient_diagnosis
        self.df_detectron2_results = df_results
        return self.df_detectron2_results


    def create_detectron2_results_figure_idx(self, idx: int):
        img = self.all_inputs[idx]
        d = self.detectron2_outputs[idx]['instances'].to('cpu')
        visualizer1 = Visualizer(
            np.stack([img[:, :, 0]]*3, axis=-1),
            metadata=MetadataCatalog.get("__nonexist__").set(thing_classes=['sane', 'sick']), scale=0.5,
        )
        visualizer2 = Visualizer(
            np.stack([img[:, :, 2]]*3, axis=-1),
            metadata=MetadataCatalog.get("__nonexist__").set(thing_classes=['sane', 'sick']), scale=0.5,
        )
        out1 = visualizer1.draw_instance_predictions(d)
        out2 = visualizer2.draw_instance_predictions(d)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(out1.get_image())
        axs[1].imshow(out2.get_image())
        return fig

    def save_detectron2_outputs(self, path, verbose=True, save_figs=False):
        pathlib.Path(join(path, 'images')).mkdir(exist_ok=True, parents=True)
        df_results = self.create_df_detectron2_outputs()
        df_results.to_csv(join(path, 'detectron2_outputs.csv'))

        self.is_positive = df_results['patient_diagnosis'].iloc[0]

        if verbose:
            self.log_console(
                "Patient Diagnosis:",
                "mri positive for AxSpA" if df_results['patient_diagnosis'].iloc[0] else "mri negative for AxSpA"
            )
            self.log_console('Saved detectron2 outputs in', join(path, 'detectron2_outputs.csv'))

        for idx in tqdm(range(len(self.mri_stir))):
            fig = self.create_detectron2_results_figure_idx(idx)
            fig.savefig(join(path, "images", f"{idx}.png"))
            plt.close(fig)
        return df_results

    @staticmethod
    def get_volume_from_mri(mri, axis_to_stack=0):
        """
        Stacks all the pixel_array of the lines of mri to form a 3D cube.

        Args:
            mri (pd.DataFrame): dataframe containing the pixel arrays
            axis_to_stack (int, optional): Axis to stack to. Defaults to 0.

        Returns:
            ndarray: 3D cube of the slices put together.
        """
        volume = []
        for img in mri.pixel_array:
            volume.append(img)
        return np.stack(volume, axis_to_stack)

    @staticmethod
    def decision_rule_threshold(df_results, thresh=2):
        x = np.array(
            df_results['label_pred_left'].sum() +
            df_results['label_pred_right'].sum()
        )
        return x[x != -1].sum() >= thresh




class PatientDicom(Patient):
    """
    Class for patient from a DICOM folder.
    """

    def __init__(self, patient_id=None, path_t1=None, path_stir=None):
        super().__init__(patient_id, path_t1=path_t1, path_stir=path_stir)
        self.path_t1 = path_t1
        self.path_stir = path_stir

    @staticmethod
    def _get_mris(path_t1, path_stir, *args, **kwargs):
        all_t1s = [join(path_t1, fil) for fil in os.listdir(path_t1)]
        mri_t1 = get_mri(all_t1s)
        mri_stir = None

        if path_stir is not None:
            all_stirs = [join(path_stir, fil) for fil in os.listdir(path_stir)]

            mri_stir = get_mri(all_stirs)

            shape_t1 = mri_t1.pixel_array.iloc[0].shape

            if shape_t1 != mri_stir.pixel_array.iloc[0].shape:
                for idx_img, stir_img in enumerate(mri_stir.pixel_array):
                    mri_stir.pixel_array.iloc[idx_img] = cv2.resize(stir_img, shape_t1, cv2.INTER_LINEAR)

        return mri_t1, mri_stir
