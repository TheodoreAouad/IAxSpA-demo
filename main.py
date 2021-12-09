import argparse
import os
from os.path import join
from time import time

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


from patient import PatientDicom
from preprocessing import ComposeProcessColumn, ConstantPixelSpacing, MinMaxNorm, RemoveGaussianBias, LocalMedian, \
    EqualizeAdaptHist


start = time()


def get_args():

    default_path_weights = "weights/weights_mrcnn.pt"
    default_patients = "data/"

    default_output = "outputs/mask_rcnn"

    parser = argparse.ArgumentParser()
    parser.add_argument('-weights', '--path_weights', type=str, help='path to weights',
                        default=default_path_weights)
    # parser.add_argument('-t1', '--path_t1', help='path to DICOM T1 sequence')
    # parser.add_argument('-stir', '--path_stir', help='path to DICOM STIR sequence')
    # parser.add_argument('-id', '--id', help='ID of the patient')
    parser.add_argument('-patients', '--all_patients', help='path to folder containing all patients', default=default_patients)
    parser.add_argument('-output', '--output_path', help='output path to save all results', default=default_output)

    return parser.parse_args()


cli_args = get_args()


channels = [0, 1, 2]
PIXEL_SPACING = 0.5
preprocessing = ComposeProcessColumn([
    ConstantPixelSpacing(PIXEL_SPACING, interpolation_input=[[1], [1], [1]]),
    MinMaxNorm(background=0, channels=channels),
    RemoveGaussianBias(coef_smoothing=.08, channels=channels),
    MinMaxNorm(background=0, channels=channels),
    LocalMedian(background=0, channels=channels),
    EqualizeAdaptHist(channels=channels),
])


def get_predictor(path_to_weights):
    cfg_eval = get_cfg()
    cfg_eval.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg_eval.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg_eval.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg_eval.INPUT.MASK_FORMAT = "bitmask"
    cfg_eval.MODEL.WEIGHTS = path_to_weights
    cfg_eval.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    return DefaultPredictor(cfg_eval)


all_patients = os.listdir(cli_args.all_patients)
for idx, patient_folder in enumerate(all_patients):
    print("=============================")
    print(f"Computing patient {idx + 1} / {len(all_patients)}.")
    patient = PatientDicom(
        patient_id=patient_folder,
        path_t1=join(cli_args.all_patients, patient_folder, 'T1'),
        path_stir=join(cli_args.all_patients, patient_folder, 'STIR'),
    )

    output_path = join(cli_args.output_path, patient_folder)

    print(f"Applying mask-rcnn on {patient_folder}...")
    patient.apply_detectron2(
        get_predictor(cli_args.path_weights),
        preprocessing=preprocessing,
        t1_chans=[2],
        stir_chans=[0, 1],
        save_inputs_path=join(output_path, "inputs"),
    )

    print(f"Saving results in {join(output_path, 'outputs')} ...")
    patient.save_detectron2_outputs(join(output_path, "outputs"))
    print("Done.")

print()
print(f'All done in {time() - start:.0f} s.')
