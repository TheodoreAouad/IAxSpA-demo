import argparse
from os.path import join

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


from patient import PatientDicom
from preprocessing import ComposeProcessColumn, ConstantPixelSpacing, MinMaxNorm, RemoveGaussianBias, LocalMedian, \
    EqualizeAdaptHist



def get_args():

    default_path_weights = "weights/weights_mrcnn.pt"
    default_t1 = "data/Patient1/T1"
    default_stir = "data/Patient1/STIR"
    default_output = "outputs/mask_rcnn"

    parser = argparse.ArgumentParser()
    parser.add_argument('-weights', '--path_weights', type=str, help='path to weights',
                        default=default_path_weights)
    parser.add_argument('-t1', '--path_t1', help='path to DICOM T1 sequence', default=default_t1)
    parser.add_argument('-stir', '--path_stir', help='path to DICOM STIR sequence', default=default_stir)
    parser.add_argument('-id', '--id', help='ID of the patient')
    parser.add_argument('-patients', '--all_patients', help='path to folder containing all patients')
    parser.add_argument('-output', '--output_path', help='output path to save all results', default=default_output)

    return parser.parse_args()


cli_args = get_args()
patient = PatientDicom(patient_id=cli_args.id, path_t1=cli_args.path_t1, path_stir=cli_args.path_stir)

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


print("Applying mask-rcnn...")
patient.apply_detectron2(
    get_predictor(cli_args.path_weights),
    preprocessing=preprocessing,
    t1_chans=[2],
    stir_chans=[0, 1],
    save_inputs_path=join(cli_args.output_path, "inputs"),
)

print("Saving results...")
patient.save_detectron2_outputs(join(cli_args.output_path, "outputs"))
print("Done.")
