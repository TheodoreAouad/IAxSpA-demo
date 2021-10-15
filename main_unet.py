import argparse
from os.path import join
import yaml
import pathlib

from torchvision.transforms import Compose, ToTensor
import torch

from patient import PatientDicom
from preprocessing import ComposeProcessColumn, MinMaxNorm, RemoveGaussianBias, LocalMedian, \
    EqualizeAdaptHist, ToUint8, TorchMinMaxNorm
from models import UNet, UNetCombine
from utils import load_yaml



def get_args():
    default_path_weights_iliac = "weights/weights_iliac.pt"
    default_path_weights_sacrum = "weights/weights_sacrum.pt"
    default_path_t1 = "data/Patient1/T1"
    default_path_stir = "data/Patient1/STIR"
    default_output_path = "outputs/segmentation"
    default_device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('-weights_iliac', '--path_weights_iliac', type=str, help='path to weights',
                        default=default_path_weights_iliac)
    parser.add_argument('-weights_sacrum', '--path_weights_sacrum', type=str, help='path to weights',
                        default=default_path_weights_sacrum)
    parser.add_argument('-t1', '--path_t1', help='path to DICOM T1 sequence', default=default_path_t1)
    parser.add_argument('-stir', '--path_stir', help='path to DICOM STIR sequence', default=default_path_stir)
    parser.add_argument('-id', '--id', help='ID of the patient')
    parser.add_argument('-patients', '--all_patients', help='path to folder containing all patients')
    parser.add_argument('-output', '--output_path', help='output path to save all results', default=default_output_path)
    parser.add_argument('-device', '--device', help='output path to save all results', default=default_device)

    return parser.parse_args()


cli_args = get_args()

if cli_args.device.lower() in ['gpu', 'cuda'] and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Computing on {device}.')

patient = PatientDicom(patient_id=cli_args.id, path_t1=cli_args.path_t1, path_stir=cli_args.path_stir)

preprocessing = ComposeProcessColumn([
    MinMaxNorm(),
    RemoveGaussianBias(coef_smoothing=.05,),
    MinMaxNorm(),
    LocalMedian(),
    EqualizeAdaptHist(),
    ToUint8(),
])

batch_preprocessing = Compose([ToTensor(), TorchMinMaxNorm()])

model_iliac = UNet(in_channels=1, n_classes=2,)
print('Loading UNet Iliac...')
model_iliac.load_state_dict(torch.load(cli_args.path_weights_iliac))

print('Loading UNet Sacrum...')
model_sacrum = UNet(in_channels=1, n_classes=2,)
model_sacrum.load_state_dict(torch.load(cli_args.path_weights_sacrum))

print('Combining both models...')
model_segm = UNetCombine(model_sacrum, model_iliac)
print('Segmentation model loaded.')


print("Applying U-Net...")
patient.segment_volume(segm_type="dl", segm_args={
    'preprocessing': preprocessing,
    'model': model_segm,
    'batch_size': 1,
    'batch_preprocessing': batch_preprocessing,
    'device': device,
})

print("Saving segmentation...")
pathlib.Path(join(cli_args.output_path, "segmentation")).mkdir(exist_ok=True, parents=True)
patient.save_segmentation(join(cli_args.output_path, 'segmentation'))


print("Roi cropping...")
args_roi = load_yaml("config/args_roi.yaml")
patient.get_rois(delete_wings_args=args_roi["DELETE_WINGS"], cropper_args=args_roi["CROPPER_ARGS"])

print("Saving rois...")
pathlib.Path(join(cli_args.output_path, "rois")).mkdir(exist_ok=True, parents=True)
patient.save_rois(join(cli_args.output_path, 'rois'))

print("Done.")
