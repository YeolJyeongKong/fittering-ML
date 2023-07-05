from torchvision import transforms
from utils.preprocessing import *

transform = transforms.Compose([
            transforms.Lambda(binary_labels_torch), 
            transforms.Lambda(crop_true),
            transforms.ToPILImage(),
            transforms.Resize((512, 512)), 
            transforms.ToTensor(),
            transforms.Lambda(convert_multiclass_to_binary_labels_torch)
        ])