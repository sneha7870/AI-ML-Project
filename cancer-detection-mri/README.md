# Cancer Detection using MRI Images (Brain Tumor Classification)

Classifies brain MRI scans by tumor type using transfer learning (VGG16 backbone, fine-tuned) in TensorFlow/Keras.

## Dataset
Uses the **Brain Tumor MRI Dataset** structure (Kaggle: `masoudnickparvar/brain-tumor-mri-dataset`), 4 classes: `glioma`, `meningioma`, `pituitary`, `notumor`. Download and arrange as:

```
data/
  Training/
    glioma/
    meningioma/
    pituitary/
    notumor/
  Testing/
    glioma/
    meningioma/
    pituitary/
    notumor/
```

Works equally well for a binary tumor/no-tumor dataset — just use 2 subfolders instead of 4; the script auto-detects the number of classes from the folder structure.

## Approach
- **Transfer learning**: VGG16 pretrained on ImageNet as a frozen feature extractor, custom dense head on top.
- **Phase 1**: train only the new head (backbone frozen) — fast convergence, prevents destroying pretrained features early.
- **Phase 2**: unfreeze the last 4 layers of VGG16 and fine-tune at a much lower learning rate (1e-5) — squeezes out extra accuracy by adapting high-level filters to MRI textures.
- **Augmentation**: rotation/shift/shear/zoom/flip — medical imaging datasets are usually small, so this matters a lot.

## Run
```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
python train.py
```
GPU recommended (VGG16 fine-tuning on CPU is slow). On Colab GPU, full training (both phases) takes roughly 20-30 minutes depending on dataset size.

## Outputs
- `sample_mri_images.png`, `training_curves.png`, `confusion_matrix.png`
- `best_mri_model.keras`, `mri_cancer_detection_final.keras`, `class_names.txt`

## Expected performance
This setup typically reaches **~96-98% test accuracy** on the 4-class Kaggle brain tumor dataset — it's a fairly clean, well-separated dataset. For your report, make sure to discuss precision/recall per class (not just accuracy) since misclassifying a tumor as `notumor` is a much costlier error than the reverse — the classification report printed at the end gives you that breakdown directly.

## Important note
This is a training exercise / demo project, not a diagnostic tool — worth stating explicitly in your report/README for academic submissions, since it's medical imaging.
