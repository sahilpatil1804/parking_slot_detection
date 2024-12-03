from utils.model import *
from utils.dataloader import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import cv2
import cvzone
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),  # Converts the image to a tensor of shape (C, H, W)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the tensor
])

# Load the trained model
model = mAlexNet().to(device)
model.load_state_dict(torch.load('model.pth'))  # Load trained weights
model.eval()  # Set the model to evaluation mode

# Open the image file
img = Image.open("imgfile3.jpg")

# Apply transformations (resize, crop, convert to tensor, normalize)
transformedimg = transform(img)

# Add a batch dimension (since models expect a batch of images, not a single image)
transformedimg = transformedimg.unsqueeze(0).to(device)  # Shape: [1, C, H, W]

# Perform inference without calculating gradients
with torch.no_grad():
    output = model(transformedimg)  # Pass the image through the model
    _, predicted = torch.max(output.data, 1)  # Get the predicted class index

# Check the prediction and print result
if predicted == 1:
    print("Car is there")
else:
    print("Car is not there")
    