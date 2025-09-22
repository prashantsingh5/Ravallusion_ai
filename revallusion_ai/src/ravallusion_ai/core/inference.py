import os
import cv2
import torch
import shutil
import torch.nn as nn
from PIL import Image

import torch.nn.functional as F
from torchvision import models, transforms

from ultralytics import YOLO

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class font_classifier_architecture(nn.Module):

    def __init__(self, num_classes=2, pretrained=True):
        super(font_classifier_architecture, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class FontClassifier:
    
    BASE_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "best_font_classifier.pth")
    
    def __init__(self, model_path : str = BASE_MODEL_PATH, device : str = "cpu"):
       
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Bad', 'Good']  
        
        self.model = self._load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 128)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # print(f"Model loaded successfully on {self.device}")

    def _load_model(self, model_path):
      
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        model = font_classifier_architecture(num_classes=len(self.class_names), pretrained=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()  
        
        return model
    
    def preprocess_image(self, image_path):
       
        try:
            image = Image.open(image_path).convert('RGB')
            
            image_tensor = self.transform(image).unsqueeze(0)  
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image {image_path}: {str(e)}")
    
    def predict_single(self, image_path):
       
        image_tensor = self.preprocess_image(image_path)
        
        with torch.inference_mode():

            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            
            result = {
                "class" : predicted_class,
                "class_idx" : predicted.item(),
                'img_path': image_path,
                'conf': confidence_score,

            }
            
            
            return result
    
    


class ObjectsDetector:

    BASE_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "best.pt")

    def __init__(self, saved_model_path : str = BASE_MODEL_PATH,
                       save_dir :str = "test",
                       debug : bool = False):
        
        torch.manual_seed(42) 

        self.model = YOLO(saved_model_path)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model.to(self.device)

        self.save_dir = save_dir
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        
        os.makedirs(self.save_dir)

        self.debug = debug


    @property
    def get_labels(self):
        return self.model.names


    def predict(self,image_path : str):

        pred = self.model.predict(
            source=image_path,
            verbose=False,   
            device=self.device
        )
        img_basename = os.path.basename(image_path)

        if self.debug:
            annotated_frame =pred[0].plot()
            cv2.imwrite(os.path.join(self.save_dir,f'pred_{img_basename}'),annotated_frame)

        results = []

        for itr,box in enumerate(pred[0].boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            results.append({"id":itr,"img_path":image_path,"class":self.get_labels[cls_id],"conf":conf,"bbox":xyxy})

        return results
    
    def predict_and_crop(self, image_path: str, temp_dir: str):
      
        results = []
        try:
            original_image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return results
        img_basename = os.path.basename(image_path)
        pred = self.model.predict(source=image_path, conf=0.7, save=False, show=False)
        if self.debug:
            annotated_frame =pred[0].plot()
            cv2.imwrite(os.path.join(self.save_dir,f'pred_{img_basename}'),annotated_frame)
        
        for itr, box in enumerate(pred[0].boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            xyxy_int = xyxy.astype(int)

            try:
                cropped_image = original_image.crop(xyxy_int)
                class_name = self.get_labels[cls_id]
                file_name = f"{class_name}_{itr}.jpg"
                save_path = os.path.join(temp_dir, file_name)
                cropped_image.save(save_path)

                results.append({
                    "id": itr,
                    "class": class_name,
                    "conf": conf,
                    "bbox": xyxy,
                    "cropped_image_path": save_path
                })
            except Exception as e:
                print(f"Error cropping or saving image for box {itr}: {e}")
                continue

        return results
    


class ImageComparator:
    def __init__(self, embedding_dim: int = 1024, device: str = "cpu"):
        self.device = device
        
        resnet = models.resnet18(pretrained=True)
        in_features = resnet.fc.in_features  # DEFAULT : 512 resnet18
        resnet.fc = nn.Linear(in_features, embedding_dim)  
        self.model = resnet
        self.model.eval().to(device)
        torch.manual_seed(42)  

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @staticmethod
    def load_and_convert(image_path: str, resize_shape: Tuple[int, int]):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        if len(image.shape) == 2:  # GRAY
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # BGRA -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:  # BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, resize_shape,interpolation=cv2.INTER_AREA)
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        image = image.permute(2, 0, 1) 
        return image

    def get_image_tensor(self, image_path: str, resize_shape: Tuple[int, int]):
        img = self.load_and_convert(image_path, resize_shape)
        img = self.normalize(img) 
        return img.unsqueeze(0).to(self.device)

    def get_similarity_score(self, image_1_path: str, image_2_path: str):
        img1 = self.get_image_tensor(image_1_path, (300, 300))
        img2 = self.get_image_tensor(image_2_path, (300, 300))

        with torch.inference_mode():
            emb1 = self.model(img1)
            emb2 = self.model(img2)

        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        return F.cosine_similarity(emb1, emb2).item()
