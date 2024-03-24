
import os
import pathlib
import sys
from tempfile import TemporaryDirectory
import numpy as np
import cv2
import torch 
from glob import glob
from PIL import Image
from retinaface import RetinaFace
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets, models
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Subset
import time
from sklearn.model_selection import train_test_split
cudnn.benchmark = True


transforms = {
    'train':v2.Compose([
            v2.ToTensor(),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val':
    v2.Compose([
            v2.ToTensor(),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

}



class StrokeDetectorClassifier():

    
    def __init__(self):
        self.model = resnet18(weights='IMAGENET1K_V1')
        
        
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.dataloders = None


    def train_model(self,criterion=torch.nn.CrossEntropyLoss(), num_epochs=25 ):
        model=self.model
        
        criterion=torch.nn.CrossEntropyLoss()
        optimizer=optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=7)
        model = self.model
        dataloaders = self.dataloders
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        model = model.to(self.device)
        since = time.time()

        # Create a temporary directory to save training checkpoints
        os.makedirs('model', exist_ok=True)
        best_model_params_path = os.path.join('model', 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
        return model
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, mode_dict_path, image: Image ):
        model = self.model
        with open(mode_dict_path, 'rb') as f:
            self.model.load_state_dict(torch.load(f ) )
        was_training = model.training
        model.eval()

        img = transforms['val'](image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)

            pred_class= self.class_names[preds[0]]
            model.train(was_training)
            return pred_class
            

    def preprocess(self, path_to_images_folder: str):
        
       
        image_files = glob(os.path.join(path_to_images_folder, '*.jpeg') ) +\
                        glob(os.path.join(path_to_images_folder, '*.jpg') ) +\
                    glob(os.path.join(path_to_images_folder, '*.png') ) +\
                    glob(os.path.join(path_to_images_folder, '*.gif') )
        # image_files = [file for file in os.listdir(path_to_images_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
        images_data = []
        for image_file in image_files:
            image = Image.open(image_file)
            image = image.convert('RGB')
            
            images_data.append(image)

        transformed_data = [transforms(image) for image in images_data]

        print(transformed_data[0].shape)

        

        return images_data
        
    def preprocess_data(self, data_dir: str):
        image_datasets = self.split_train_val_data(data_dir)
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].dataset.classes


        self.dataloders = dataloaders
        self.class_names = class_names
        self.dataset_sizes = dataset_sizes

    def split_train_val_data(self, data_dir: str, val_split: float = 0.25):
        dataset = ImageFolder(os.path.join(data_dir),transform=transforms['train'] )  

        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['val'] = Subset(dataset, val_idx)
        return datasets

        