import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.models import resnet50

checkpoint_path = "../asl/checkpoint_data_gen.pt"

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),# Resize images to a fixed size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the images
])


class Resnet50_Fine(torch.nn.Module):
    def __init__(self,num_classes):
        super(Resnet50_Fine, self).__init__()

        self.pretrained = resnet50(pretrained=True)

        num_ftrs = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Linear(num_ftrs, num_classes)
    def forward(self, x):
        x = self.pretrained(x)
        return x




def load_model(model):
    saved_state_dict = torch.load(checkpoint_path)
    model.load_state_dict(saved_state_dict)
    return model