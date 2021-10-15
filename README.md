# IAxSpA-demo

## Repository Description
This is a demo of the automatic detection of axSpA using deep learning.

Our method will be described in more details in an upcoming paper.

This code can be used :
- to segment the iliac and sacrum in semi-coronal pelvic MRIs
- to recover the Region of Interest (ROI) for diagnosing axial ankylosing spondylitis
- to assess the presence or absence of axial ankylosing spondylitis on the MRI of a patient with T1 and STIR sequences.

On this repository, you can find on the "release" tab three important weights files:
- `weights/weights_iliac.pt`: the weights for the U-Net that predicts the iliac
- `weights/weights_sacrum.pt`: the weights for the U-Net that predicts the sacrum
- `weights/weights_mrcnn.pt`: the weights for the Mask-RCNN that predicts the inflammation

We give the STIR and T1 of a real patient as an example in `data/Patient1`.


## Requirements

This code was tested on an Ubuntu 18.04 system. For now, you will need a functioning cuda environment such that the following command does not return an error: `import torch; torch.ones(1).to('cuda')`

Works for Python version >=3.7. The following python libraries are necessary:

- numpy
- pandas
- torch
- torchvision
- scipy
- opencv-python
- scikit-image
- pydicom
- pyyaml
- detectron2
- tqdm
- matplotlib
- Pillow



## Launch the code


### U-Net segmentation of ROI

To use the U-Net to segment the iliac / sacrum and predict the ROIs (replace paths between `{}` by real paths):

`python main_unet -weights_iliac {path/to/weights_iliac.pt} -weights_sacrum {path/to/weights_sacrum.pt} -t1 data/Patient1/T1 -stir data/Patient1/STIR -output {path/to/output} -output {path/to/output} -device cuda`

You can also choose to run on cpu, with `-device cpu`


In the `{path/to/output}` folder:
- a folder `segmentation` will be created where the segmentations of the MRIs will be saved
- a folder `roi` will be created where the ROIs of the MRIs will be saved

### Mask-RCNN diagnosis

To use the Mask-RCNN to segment the ROI, classify the ROIs and diagnose the patient (replace paths between `{}` by real paths):

`python main -weights {path/to/weights_mrcnn.pt} -t1 data/Patient1/T1 -stir data/Patient1/STIR -output {path/to/output}` 

