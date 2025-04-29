#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# We are ready to train the Cardiac Detection Model now!

# ## Imports:
# 
# * torch and torchvision for model and dataloader creation
# * pytorch lightning for efficient and easy training implementation
# * ModelCheckpoint and TensorboardLogger for checkpoint saving and logging
# * numpy data loading
# * cv2 for drawing rectangles on images
# * imgaug for augmentation pipeline
# * Our CardiacDataset
# 
# 

# In[2]:


import warnings
warnings.filterwarnings("ignore", message="Checkpoint directory.*exists and is not empty")
from dataset import CardiacDataset
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import cv2
import imgaug.augmenters as iaa
from torchvision.models import ResNet18_Weights
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]


# We create the dataset objects and the augmentation parameters to specify the augmentation parameters

# In[3]:


train_root_path = BASE_DIR / "data" / "processed" / "train"
train_subjects = BASE_DIR / "data" / "processed" / "train_subjects_det.npy"
val_root_path = BASE_DIR / "data" / "processed" / "val"
val_subjects = BASE_DIR / "data" / "processed" / "val_subjects_det.npy"

train_transforms = iaa.Sequential([
                                iaa.GammaContrast(),
                                iaa.Affine(
                                    scale=(0.8, 1.2),
                                    rotate=(-10, 10),
                                    translate_px=(-10, 10)
                                )
                            ])


# In[4]:


train_dataset = CardiacDataset(BASE_DIR / "resources" / "rsna_heart_detection.csv", train_subjects, train_root_path, train_transforms)
val_dataset = CardiacDataset(BASE_DIR / "resources" / "rsna_heart_detection.csv", val_subjects, val_root_path, None)



# In[ ]:


import os
import sys


# In[ ]:


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


# In[ ]:


# Use more workers if safe (e.g. script, not notebook)
if os.name == 'nt' and not is_notebook():
    num_workers = min(8, os.cpu_count())  # Use up to 8 workers safely
    persistent_workers = True
else:
    num_workers = 0
    persistent_workers = False


# Adapt batch size and num_workers according to your computing hardware.

# In[5]:


batch_size = 8

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


# ## Model Creation

# We use the same architecture as we used in the classifcation task with some small adaptations:
# 
# 1. 4 outputs: Instead of predicting a binary label we need to estimate the location of the heart (xmin, ymin, xmax, ymax).
# 2. Loss function: Instead of using a cross entropy loss, we are going to use the L2 loss (Mean Squared Error), as we are dealing with continuous values.

# In[17]:


class CardiacDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Load pretrained ResNet18 and adapt it
        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=4)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.MSELoss()

        # Tracking losses per epoch
        self.train_losses = []
        self.val_losses = []

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)
        loss = self.loss_fn(pred, label)
        self.log("train_loss", loss)

        if batch_idx % 50 == 0:
            self.log_images(x_ray.cpu(), pred.cpu(), label.cpu(), "Train")

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss, prog_bar=True)
        self.train_losses.append(avg_loss.item())

    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)
        loss = self.loss_fn(pred, label)
        self.log("val_loss", loss)

        if batch_idx % 50 == 0:
            self.log_images(x_ray.cpu(), pred.cpu(), label.cpu(), "Val")

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss, prog_bar=True)
        self.val_losses.append(avg_loss.item())

    def log_images(self, x_ray, pred, label, name):
        results = []
        for i in range(min(4, x_ray.size(0))):  # Protect against batches smaller than 4
            coords_labels = label[i]
            coords_pred = pred[i]
            img = ((x_ray[i] * 0.252) + 0.494).numpy()[0]

            # Draw ground truth bbox
            x0, y0 = coords_labels[0].int().item(), coords_labels[1].int().item()
            x1, y1 = coords_labels[2].int().item(), coords_labels[3].int().item()
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)

            # Draw predicted bbox
            x0, y0 = coords_pred[0].int().item(), coords_pred[1].int().item()
            x1, y1 = coords_pred[2].int().item(), coords_pred[3].int().item()
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (1, 1, 1), 2)

            results.append(torch.tensor(img).unsqueeze(0))

        grid = torchvision.utils.make_grid(results, 2)
        self.logger.experiment.add_image(f"{name} Prediction vs Label", grid, self.global_step)

    def configure_optimizers(self):
        return [self.optimizer]

    def save_metrics(self, filename="training_metrics.csv"):
    	import pandas as pd

    	# Ensure all lists have the same length
    	min_len = min(len(self.train_losses), len(self.val_losses))

    	df = pd.DataFrame({
        	"epoch": list(range(min_len)),
        	"train_loss": self.train_losses[:min_len],
        	"val_loss": self.val_losses[:min_len]
    	})
    	df.to_csv(filename, index=False)


# In[18]:


# Create the model object
if __name__ == "__main__":

	checkpoint_callback = ModelCheckpoint(
    		monitor="avg_val_loss",        # Monitorea la pérdida de validación promedio
    		mode="min",                    # Queremos minimizar la val_loss
    		save_top_k=1,                  # Solo guarda el mejor modelo
    		dirpath=BASE_DIR / "models",   # Carpeta donde guardar el modelo
    		filename="best-cardiac-detection",  # Nombre base del archivo
    		save_last=True                 # Opcional: también guarda el último modelo
	)


	# Train for at least 50 epochs to get a decent result.
	# 100 epochs lead to great results.
	# 
	# You can train this on a CPU!
	# Create the trainer
	# Change the gpus parameter to the number of available gpus in your computer. Use 0 for CPU training

	trainer = pl.Trainer(
    		gpus=1,
    		logger=TensorBoardLogger(BASE_DIR / "logs", name="cardiac_detection"),
    		default_root_dir=BASE_DIR / "models",
    		callbacks=[checkpoint_callback],
    		max_epochs=100
	)
	model = CardiacDetectionModel().to("cuda")  # Instanciate the model

	# Train the detection model
	trainer.fit(model, train_loader, val_loader)
	model.save_metrics(BASE_DIR / "models" / "training_metrics.csv")

