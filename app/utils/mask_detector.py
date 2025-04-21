import torch
from torchvision import transforms
from PIL import Image
from app.model.mask_model import MaskModel

class MaskDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MaskModel.load_model(model_path, self.device)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.labels = {0: "No Mask", 1: "Mask", 2: "Incorrect Mask"}

    def predict(self, image_file):
        try:
            image = Image.open(image_file).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                _, pred = torch.max(outputs, 1)
                
            return {
                "label": self.labels[int(pred)],
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

# Initialize detector instance
detector = MaskDetector('app/model/mask_detector.pth')
