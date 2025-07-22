import os

class Config:
    def __init__(self):
        self.DATASET_PATH = "/kaggle/working/fer2013"
        self.MODEL_NAME = "EfficientNet"
        self.NUM_CLASSES = 4
        self.BATCH_SIZE = 128 
        self.NUM_EPOCHS = 30
        self.LEARNING_RATE = 1e-3  
        self.CRITERION = 'cross_entropy'  # Start simple, then experiment
        self.SEED = 42
        self.CHECKPOINT_PATH = "/kaggle/working/checkpoints"
        self.BEST_MODEL_PATH = os.path.join(self.CHECKPOINT_PATH, "best_model.pth")
        self.FINAL_MODEL_PATH = os.path.join(self.CHECKPOINT_PATH, "final_model.pth")
        self.USE_TQDM = True
        self.EARLY_STOP_PATIENCE = 15  # More aggressive early stopping
        self.WEIGHT_DECAY = 1e-3  # Increased L2 regularization
        self.IMAGE_SIZE = (128, 128)
        
        # New anti-overfitting parameters
        self.DROPOUT_RATE = 0.6  # Increased dropout
        self.LABEL_SMOOTHING = 0.1  # Add label smoothing
        self.GRADIENT_CLIPPING = 1.0  # Gradient clipping to prevent exploding gradients
        self.MIN_LR = 1e-7  # Minimum learning rate for scheduler
        
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)
