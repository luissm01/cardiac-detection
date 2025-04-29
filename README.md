# 🫀 Cardiac Detection from Chest X-Rays

This project focuses on training a deep learning model to detect the heart's location in chest X-ray images by predicting bounding box coordinates. It includes data preprocessing, model training using PyTorch Lightning, visual evaluation, and TensorBoard integration.

---

## 📁 Project Structure

```
cardiac-detection/
│
├── notebooks/           # Jupyter notebook for training and evaluation
│   ├── preprocess.ipynb
│   ├── dataset.ipynb
│   └── train.ipynb
│
├── scripts/             # Python scripts for training and dataset handling
│   ├── train.py
│   └── dataset.py
│
├── data/                # Input and preprocessed data (not included in Git)
│
├── models/              # Saved checkpoints (.ckpt)
│
├── logs/                # TensorBoard logs
│
├── resources/           # CSV label file
│
├── README.md            # Project overview
├── requirements.txt     # Basic dependencies
├── environment.yml      # Full environment for conda
└── .gitignore
```

---

## 🔧 How to Use

1. Clone the repository:
```bash
git clone https://github.com/luissm01/cardiac-detection.git
cd cardiac-detection
```

2. Create a virtual environment and install dependencies:

```bash
# With pip
pip install -r requirements.txt

# Or with conda
conda env create -f environment.yml
conda activate pytorchenv
```

3. Train the model:
```bash
python scripts/train.py
```

4. Open the notebook and evaluate predictions:
```bash
notebooks/cardiac_detection.ipynb
```

5. Launch TensorBoard to visualize losses and predictions:
```bash
tensorboard --logdir=logs/cardiac_detection
```

---

## 📦 Data Preparation

This project uses images derived from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge), repurposed to locate the heart instead of detecting pneumonia.

To prepare the data:
1. Download the DICOM images from the Kaggle challenge (requires login and acceptance of terms).
2. Place the images in `data/raw/rsna-heart_detection/stage_2_train_images/`.
3. Run the preprocessing notebook:
```bash
notebooks/cardiac_detection.ipynb  # 01-Preprocess section
```

The notebook will resize, normalize, and convert the images into `.npy` format, organizing them for training and validation.

---

## 📊 Model Evaluation

The model is trained to predict bounding boxes `[x0, y0, x1, y1]` corresponding to the top-left and bottom-right corners of the heart.

Loss curves and bounding box overlays are logged to TensorBoard. You can visually verify the accuracy of predictions and compare them to ground truth annotations.

---

## 👤 Author

Luis Sánchez Moreno – Biomedical Engineer specialized in AI for medical imaging.