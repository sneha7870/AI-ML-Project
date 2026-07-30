# CIFAR-10 Image Classification using CNN

Classifies 32x32 color images into 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using a custom CNN built in TensorFlow/Keras.

## Architecture
3 convolutional blocks (32 → 64 → 128 filters), each with double Conv2D + BatchNorm + MaxPooling + Dropout, followed by a dense classification head. Uses data augmentation (rotation, shift, flip, zoom) since CIFAR-10 is small (50K training images) and prone to overfitting.

## Run
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
python train.py
```
`tf.keras.datasets.cifar10.load_data()` auto-downloads the dataset (~170MB) on first run — needs internet access. **GPU strongly recommended** (50 epochs on CPU can take hours; on a Colab GPU it's ~15-20 min).

## Outputs
- `sample_images.png` — preview of training samples
- `training_curves.png` — accuracy/loss over epochs
- `confusion_matrix.png` — per-class performance
- `best_cifar10_model.keras`, `cifar10_cnn_final.keras`

## Expected performance
This architecture typically reaches **~85-88% test accuracy** after ~40-50 epochs with augmentation. To push higher (90%+), consider transfer learning (ResNet/EfficientNet pretrained on ImageNet, fine-tuned) or a deeper ResNet-style architecture with skip connections.

## Tips for your submission
- Run in Google Colab (free GPU: Runtime → Change runtime type → GPU) if you don't have local GPU access.
- Reduce `EPOCHS` to ~20-25 for faster iteration while testing.
