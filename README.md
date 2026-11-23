# CoLSI-Net

English | [Chinese](README-CN.md)

CoLSI-Net performs category-aware segmentation of the complete visible regions of cooperative landing surfaces. Built on the MSCAN-B backbone and LightHam decoder of SegNeXt-B, it combines **CFA (Cycle-consistent Feature Alignment)** and **QSF (Quality–Semantic Fusion)** through **CCSA (Consistency-guided Cross-scale Support Attention)**.

The complete pipeline is: **RGB image, camera intrinsics K, and camera-frame gravity g → CoLSI-Net categories and visible masks → frozen MoGe-3 ViT-L geometry descriptors and ConvNeXt-Tiny RGB-ROI features → 824-dimensional concatenation → MLP residual fusion → surface inclination and landing status**.

## Dataset download

**Baidu Netdisk:** [Download CoLSI](https://pan.baidu.com/s/1Evur78E1ZnBxYYDhKmiagg)

**Extraction code:** `2026`

Place the downloaded `CoLSI-data` folder next to the code directory, as shown below.

## Repository layout

```text
CoLSI-code/                 # This repository
  colsi/                   # Final network, standard pipeline, training, and evaluation
  configs/                 # Training protocol and camera-input example
  third_party/             # Required upstream source code and original licenses
  train.py
  predict.py
  evaluate.py
CoLSI-data/                 # Downloaded assets; place beside the code directory
  CoLSI/                   # CoLSI-Sim, RealMarks, RealCalib, and fixed splits
  weights/
    colsi_net.pt
    convnext_tiny_roi.pt
    fusion_mlp.pt
    moge3_vitl.pt
    mscan_b_imagenet.pth
    manifest.json
```

Training and evaluation support `--data-root`; all three entry points support `--weights-dir` for custom asset locations. The code occupies a few MiB, with data and model weights stored separately.

## Environment

Use Python 3.12 with CUDA-enabled PyTorch and torchvision. The tested environment uses Python `3.12.13`, PyTorch `2.12.0.dev20260408+cu128`, torchvision `0.27.0.dev20260407+cu128`, CUDA 12.8, and an RTX 5090. Key dependency versions are recorded in `environment-tested.json`. Install a CUDA-enabled torch/torchvision pair using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), then run:

```bash
python -m pip install -r requirements.txt
```

Required source code for MoGe, FlexGEMM, and utils3d-moge is included. Inference automatically adds the bundled paths. Windows uses triton-windows; Linux uses triton. FlexGEMM uses its Triton backend. Neural training and inference run on CUDA; GPU name, capacity, model/input devices, and memory usage are recorded in the output directory at startup.

## Single-image segmentation and the complete pipeline

```bash
# Segmentation: export semantic labels, object categories, and complete visible masks
python predict.py --input image.png --segmentation-only --output outputs/segmentation

# Complete pipeline: provide calibrated K and camera-frame g_C in camera.json
python predict.py --input image.png --camera camera.json --output outputs/landing
```

See `configs/camera.example.json` for the camera-file format. K uses native-image pixel coordinates, and g_C is the gravity direction in camera coordinates. The released MoGe FOV adapter requires square pixels and a principal point at `((W−1)/2,(H−1)/2)`. MoGe uses level 9, refine 3, and FP32. The example intrinsics correspond to a 640×360 image; supply calibrated values for your actual input.

Outputs include `semantic.png`, a binary visible mask for each object, `prediction.json`, and device records. Landing status is assigned from the unrounded inclination: 0°–5° is safe, >5°–15° is cautious, and >15° is unsafe. Unavailable inclination and landing-status fields are serialized as null.

## Training

```bash
python train.py --output outputs/colsi_net_seed17
python train.py --output outputs/colsi_net_seed17 --resume
```

The default configuration in `configs/train.json` uses seed 17, 90 epochs, physical batch size 24 / effective batch size 48, BF16, TF32 disabled, AdamW, a main-branch learning rate of 0.0006, a backbone learning-rate multiplier of 0.1, and a single cosine schedule with multipliers from 1 to 0.1. Supervision combines semantic CE with 0.1-weighted boundary BCE, using the original RGB/semantic preprocessing and fixed data splits. At each completed epoch, the trainer saves `last.pt` together with optimizer, scheduler, and RNG states, and selects `best.pt` by foreground mIoU on the complete validation split. Resumption starts from the latest completed epoch; results are written to a separate output directory.

Official MSCAN-B initialization weights are provided in the cloud assets for retraining. The standard downstream pipeline uses the supplied frozen MoGe, ConvNeXt, and MLP weights.

## Evaluation

```bash
python evaluate.py --task segmentation --split val --output outputs/val_seg
python evaluate.py --task segmentation --split test --output outputs/test_seg
python evaluate.py --task full-chain --split val --output outputs/val_chain
```

Evaluation follows the bundled canonical order, segmentation batch size 8, temporary NMF seed 61107, BF16, and TF32 disabled. Images are letterboxed to 288×512 with OpenCV; native-resolution probability restoration follows the original implementation. Complete-pipeline predictions are saved before ground truth is read for scoring. Spatial matching uses IoU 0.5, and joint F1 uses angular-error thresholds of 1°/2°/3°/5°/10°. Missed objects, incorrect categories, and invalid inclination estimates receive a 90° penalty in MAE90. RGB and MLP inference uses FP32 on each frame's object group, with runtime settings recorded alongside results. Single-image demos use batch size 1; formal segmentation evaluation uses batch size 8.

Existing final-model results are **98.685922% validation foreground mIoU** and **98.719127% test foreground mIoU**. The existing standard-pipeline test results are **83.263347% joint F1@5°** and **3.385475° MAE90**. CoLSI-Sim provides the primary precise supervision; RealMarks contains weak labels, and RealCalib provides unknown-checkerboard conditional diagnostics and support-surface references.

## Provenance, validation, and licensing

`SOURCE_PROVENANCE.json` records the original sources and extraction mapping of the final core functions. Extraction preserves network parameter names. CPU validation includes strict loading of 888 final-checkpoint tensors, equality of 242 tensors in the original/released small fixture, and a maximum forward difference of 0. ROI cropping, the 48-dimensional geometry descriptor, the 824-dimensional feature order, and the MLP were also checked individually. The code release uses existing weights; reported research results retain their original experiment identities.

Upstream sources and licenses are listed in `THIRD_PARTY.md` and `third_party/*/LICENSE`. The authors specify the release license for original project code; third-party components retain their respective licenses.
