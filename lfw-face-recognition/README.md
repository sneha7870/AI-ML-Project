# Face Recognition using CNN on LFW ("Labeled Faces in the Wild")

Identifies which of several well-known people is shown in a grayscale face crop, using a CNN trained on the LFW dataset.

## Approach
- **Data**: `sklearn.datasets.fetch_lfw_people(min_faces_per_person=40, resize=0.5)` — this keeps only identities with ≥40 images (so each class has enough samples to train on) and downsamples images for speed. With `min_faces_per_person=40` you typically get ~12-15 identities and ~1,800-2,200 images total.
- **Preprocessing**: normalized pixel values to [0,1], added channel dimension for grayscale CNN input.
- **Augmentation**: rotation/shift/zoom/flip — important here since LFW is small and class-imbalanced (some celebrities have far more photos than others, e.g. politicians photographed at many events).
- **Architecture**: 3-block CNN (32→64→128 filters) with BatchNorm + Dropout, dense head with softmax over identities.

## Run
```bash
pip install scikit-learn tensorflow numpy matplotlib seaborn
python train.py
```
`fetch_lfw_people` downloads (~200MB) from scikit-learn's data repository on first run — needs internet access; the file is cached locally afterward (`~/scikit_learn_data`).

## Outputs
- `sample_faces.png` — preview of training faces
- `training_curves.png`, `confusion_matrix.png`, `sample_predictions.png` (green = correct, red = wrong)
- `best_lfw_model.keras`, `lfw_face_recognition_final.keras`, `target_names.npy`

## Notes & tuning knobs
- **`min_faces_per_person`**: raise it (e.g. 70) for fewer, more balanced classes and higher accuracy; lower it (e.g. 20) for more identities but a harder, more imbalanced problem.
- Class imbalance is the main challenge here — if accuracy looks high but per-class recall in the classification report is poor for minority classes, consider `class_weight='balanced'` in `model.fit()` or oversampling.
- For genuinely higher accuracy (~95%+), transfer learning from a face-pretrained model (e.g. FaceNet embeddings + a classifier) beats a from-scratch CNN on this dataset size — happy to build that version too if you want it for comparison in your report.
