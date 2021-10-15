from typing import List
import pathlib

import pandas as pd
import numpy as np
import pydicom
from pydicom.tag import Tag


def get_mri(
    all_paths_imgs: List[str], patient_id: str = None, get_dicom: bool = False, sort: bool = True,
) -> pd.DataFrame:
    """
    Base function to read a sequence of dicoms.

    Args:
        all_paths_imgs (str): Path to the folder containing the dicom files.
        get_dicom (bool, optional): Returns the dicom in the dataframe. Costy. Defaults to False.

    Returns:
        pandas.core.frame.DataFrame: Dataframe where each row is a slice.
    """
    mri = []
    for path_img in all_paths_imgs:
        try:
            current_mri = pydicom.dcmread(path_img)
        except pydicom.errors.InvalidDicomError:
            # print('Error while opening dicom for {}'.format(path_img))
            continue

        mri.append(pd.DataFrame({
            'patient': [patient_id],
            'path': [path_img],
            'filename': [pathlib.Path(path_img).stem],
            'dicom': [current_mri if get_dicom else None],
            'pixel_array': [current_mri.pixel_array.astype(np.int16)],
            'cosdirs': [[float(k) for k in current_mri[Tag(0x00200037)].value] if current_mri.get(Tag(0x00200037)) is not None else []],
            'ImagePosition': [[float(k) for k in current_mri.get(Tag(0x00200032)).value] if current_mri.get(Tag(0x00200032)) is not None else None],
            'y_pos': [float(current_mri.get(Tag(0x00200032)).value[1]) if current_mri.get(Tag(0x00200032)) is not None else None],
            'PixelSpacing': [[float(k) for k in current_mri.get(Tag(0x00280030)).value] if current_mri.get(Tag(0x00280030)) is not None else None],
            'InstanceNumber': [int(current_mri.get(Tag(0x00200013)).value) if current_mri.get(Tag(0x00200013)) is not None else None],
            'SliceLocation': [
                float(current_mri.get(Tag(0x00201041)).value)
                if current_mri.get(Tag(0x00201041)) is not None else None],
            'SliceThickness': [
                float(current_mri.get(Tag(0x00180050)).value)
                if current_mri.get(Tag(0x00201041)) is not None else None],
            'RepetitionTime': [float(current_mri[Tag(0x00180080)].value) if current_mri.get(Tag(0x00180080)) is not None else None],
            'EchoTime': [
                float(current_mri[Tag(0x00180081)].value) if current_mri.get(Tag(0x00180081)) is not None else None],
            'ScanOptions': [
                (current_mri[Tag(0x00180022)].value) if current_mri.get(Tag(0x00180022)) is not None else None],


        }))
    mri = pd.concat(mri, sort=True).drop_duplicates(subset='y_pos')
    if sort:
        mri = mri.sort_values('y_pos')
    return mri.reset_index(drop=True)
