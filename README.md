# ViT-HyperSense: Hyperparameter Sensitivity Analysis of Vision Transformers

This repository contains code and experiments for studying the sensitivity of Vision Transformers (ViT) to various hyperparameters. The project focuses on evaluating the performance impact of parameters such as learning rate, batch size, weight decay, and training epochs on smaller ViT models like ViT-Tiny and ViT-Small. 

## **Overview**
Vision Transformers (ViT) have gained significant attention for their effectiveness in image classification and other computer vision tasks. However, their performance is often sensitive to hyperparameters, which can significantly influence training convergence and accuracy. This project aims to:
- Explore the effect of key hyperparameters on ViT performance.
- Provide insights into optimal hyperparameter settings for small-scale ViT models.
- Leverage datasets like Tiny-ImageNet or ImageNet subsets for efficient experimentation.

## **Features**
- Hyperparameter sensitivity experiments for ViT models.
- Support for custom datasets and benchmarks.
- Modular implementation using PyTorch and PyTorch Lightning for ease of experimentation.

## **Installation**
### **1. Clone the repository**
```bash
git clone https://github.com/username/ViT-HyperSense.git
cd ViT-HyperSense
```

### **2. Install dependencies**
```bash
conda create -n vit python=3.10
conda activate vit
pip install -r requirements.txt
```

### **3. Install PyTorch**
Ensure CUDA compatibility and install PyTorch:
```bash
pip install torch torchvision torchaudio
```
## **Usage**
### **1. Train a ViT Model**
To start training a ViT-Tiny model with default hyperparameters:
```bash
python train/train.py --config configs/vit_tiny_config.yaml
```

### **2. Run Hyperparameter Sensitivity Experiments**
To explore the effect of learning rate, batch size, or other hyperparameters:
```bash
python train/train.py --config configs/vit_tiny_config.yaml --learning_rate 0.001 --batch_size 64
```

### **3. Results Visualization**
After training, you can visualize the results using TensorBoard:
```bash
tensorboard --logdir results/logs/
```

## **Dependencies**
- Python 3.10
- PyTorch 2.x
- PyTorch Lightning
- timm
- TensorBoard
- Matplotlib
- Scikit-learn
- Pandas
For detailed dependencies, see requirements.txt.



## **License**
This project is open-sourced under the MIT License - see the LICENSE file for details.

## **Acknowledgments**
This project is developed as part of a research effort on Vision Transformer hyperparameter sensitivity. Special thanks to [Professor’s Name] for guidance and support.


Happy experimenting with ViT-HyperSense!