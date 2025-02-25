# Knee Injury Detection using Vision Transformers and CNN

This project aims to detect knee injuries using the MRNet dataset and various models including Vision Transformers (ViT, DeiT, Swin) and a Convolutional Neural Network (CNN). The project compares the performance of these models in classifying knee MRI images.

## Project Structure

- `data/`: Contains the dataset for training, testing, and validation.
- `models/`: Stores the trained models.
- `notebooks/`: Jupyter notebooks for experimentation and visualization.
- `src/`: Source code for data preprocessing, model building, training, and evaluation.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset in the `data/` directory.
2. Run the data preprocessing script:
    ```bash
    python src/data_preprocessing.py
    ```
3. Train the models:
    ```bash
    python src/train.py --model vit
    python src/train.py --model deit
    python src/train.py --model swin
    python src/train.py --model cnn
    ```
4. Evaluate the models:
    ```bash
    python src/evaluate.py --model vit
    python src/evaluate.py --model deit
    python src/evaluate.py --model swin
    python src/evaluate.py --model cnn
    ```

## Files

- `data_preprocessing.py`: Script for preprocessing the data.
- `train.py`: Script for training the models.
- `evaluate.py`: Script for evaluating the models.
- `vit_model.py`: Script for defining the Vision Transformer model.
- `deit_model.py`: Script for defining the DeiT model.
- `swin_model.py`: Script for defining the Swin Transformer model.
- `cnn_model.py`: Script for defining the Convolutional Neural Network model.