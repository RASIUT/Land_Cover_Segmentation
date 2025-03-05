# client.py
import requests
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
import sys
import zipfile

def download_kaggle_dataset():
    try:
        import subprocess
        import os
        
        print("Checking for Kaggle credentials...")
        kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
        if not os.path.exists(kaggle_path):
            print("Kaggle API credentials not found. Please place your kaggle.json file in ~/.kaggle/")
            return False
        
        print("Installing Kaggle package...")
        subprocess.run(['pip', 'install', '-q', 'kaggle'])
        
        print("Downloading dataset...")
        subprocess.run([
            'kaggle', 'datasets', 'download', 
            '-d', 'adrianboguszewski/landcoverai',
            '-p', '.'
        ], check=True)
        
        print("Extracting dataset...")
        # Use zipfile instead of unzip command
        with zipfile.ZipFile('landcoverai.zip', 'r') as zip_ref:
            zip_ref.extractall('landcoverai')
        
        print("Dataset downloaded and extracted successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error in subprocess: {e}")
        return False
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def prepare_image_paths():
    print("Preparing image paths...")
    image_path = Path('./landcoverai/images')
    if not image_path.exists():
        print("Dataset not found locally, attempting to download...")
        success = download_kaggle_dataset()
        if not success:
            raise Exception("Failed to download dataset")
        image_path = Path('./landcoverai/images')
    
    image_paths = list(image_path.glob('*.tif'))
    print(f"Found {len(image_paths)} images")
    return image_paths

def check_server_status():
    try:
        response = requests.get('http://localhost:5000')
        return response.status_code == 200
    except:
        return False

def send_image_to_server(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, 'rb') as image:
            response = requests.post('http://localhost:5000/segment', files={'image': image})
        
        if response.status_code != 200:
            raise Exception(f"Server error: {response.text}")
        
        output_path = str(image_path).replace('.tif', '_segmented.png')
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure server.py is running.")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def visualize_results(original_image_path, segmented_image_path):
    try:
        if not os.path.exists(original_image_path):
            raise FileNotFoundError(f"Original image not found: {original_image_path}")
        if not os.path.exists(segmented_image_path):
            raise FileNotFoundError(f"Segmented image not found: {segmented_image_path}")
            
        original_img = cv2.imread(str(original_image_path), cv2.IMREAD_COLOR)
        if original_img is None:
            raise ValueError("Failed to load original image")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        segmented_img = cv2.imread(segmented_image_path, cv2.IMREAD_COLOR)
        if segmented_img is None:
            raise ValueError("Failed to load segmented image")
        segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(original_img)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Segmented Image')
        plt.imshow(segmented_img)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error visualizing results: {str(e)}")

def main():
    try:
        print("Starting land cover segmentation client...")
        
        # Check if server is running
        print("Checking server status...")
        if not check_server_status():
            print("Warning: Could not connect to server. Make sure server.py is running on localhost:5000")
            if input("Do you want to continue anyway? (y/n): ").lower() != 'y':
                sys.exit(1)
        
        # Get image paths
        image_paths = prepare_image_paths()
        
        if not image_paths:
            raise Exception("No images found in the dataset")
        
        # Process images
        sample_size = min(3, len(image_paths))
        sample_images = random.sample(image_paths, sample_size)
        
        print(f"\nProcessing {sample_size} random images...")
        for image_path in tqdm(sample_images, desc="Processing images"):
            print(f"\nProcessing {image_path}")
            
            segmented_image_path = send_image_to_server(image_path)
            
            if segmented_image_path:
                print("Visualizing results...")
                visualize_results(image_path, segmented_image_path)
            else:
                print("Skipping visualization due to processing error")
        
        print("\nProcessing complete!")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()