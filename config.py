import os

class Config:
    MODEL_PATH = os.path.join('app', 'model', 'mask_detector.pth')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB limit