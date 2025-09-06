from model import YOLOv11mLSTMClassifier
import torch
from torchvision import transforms
import numpy as np
import cv2


class Classifier:
    
    classification_model_path = '/classification_result/weights/best_model.pth'

    def __init__(self, image_bytes):

        self.image_bytes = image_bytes
        self.device = self._get_device()
        self.model = self._load_model(self.classification_model_path)
    
    def _get_device(self):
        """
        Return the device for running the model
        """
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    async def __call__(self):
        """
        
        """
        frame = self._bytes_to_image()

        if frame is not None:
            classes = self.classify(frame)
            return classes
        else:
            raise ValueError('Frame cannot be none')
    
    def _load_model(self, path):
        """Loads classification model from the path

        Returns:
            model - Trained classification model
        """
        model = YOLOv11mLSTMClassifier(num_classes=2).to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _bytes_to_image(self):
        """
        Taking the input from API which is in bytes

        Returning a cv2 image format for processing 
        """
        array = np.asarray(bytearray(self.image_bytes), dtype=np.uint8)
        image = cv2.imdecode(array, -1)

        return image
    
    def classify(self, frame):

        class_names = ['clean', 'not clean']

        preprocess = transforms.Compose([
            transforms.ToPIL(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        input_tensor = preprocess(np.array(frame))
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
        
        predicted_label = class_names[predicted.item()]

        return predicted_label