# server.py
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, send_file
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Define model classes
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x.float())
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Flask app setup
app = Flask(__name__)

# Define classes and colors
CLASSES = ['background', 'building', 'woodland', 'water', 'road']
COLORS = {
    0: [0, 0, 0],       # background: black
    1: [255, 0, 0],     # building: red
    2: [0, 255, 0],     # woodland: green
    3: [0, 0, 255],     # water: blue
    4: [255, 255, 0]    # road: yellow
}

# Load the model
try:
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=len(CLASSES))
    state_dict = torch.load('model_data.pt', map_location=device, weights_only=False)
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    else:
        model = state_dict
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def preprocess_image(image):
    # Resize image to 256x256
    image = cv2.resize(image, (256, 256))
    
    # Convert to float and normalize
    image = image / 255.0
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
    image_tensor = normalize(image_tensor)
    
    return image_tensor

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        # Get the uploaded image file
        file = request.files['image']
        
        # Read and preprocess image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = preprocess_image(img_rgb)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)
            pred_mask = torch.argmax(output, dim=1).squeeze().numpy()
        
        # Create colored segmentation mask
        seg_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        for cls in range(len(CLASSES)):
            seg_mask[pred_mask == cls] = COLORS[cls]
        
        # Save and return
        plt.imsave('segmented_image.png', seg_mask)
        return send_file('segmented_image.png', mimetype='image/png')
    
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)