import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image, ImageFile
from albumentations.pytorch import ToTensorV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CustomDataset(Dataset):
    def __init__(self, file_path,imagesize):
        self.dir=file_path
        self.data =os.listdir(file_path)
        self.img=imagesize
        self.trans=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((self.img, self.img)),
        #for traion
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ])
        # self.trans=A.Compose(
        #         [
        #     A.Resize(height=IMG_SIZE, width=IMG_SIZE),

        #     ]
        #     )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample=os.path.join(self.dir,sample)
        img = np.array(Image.open(sample).convert("RGB"))
        img=self.trans(img)
        return img

class DatasetLoader(DataLoader):
    def __init__(self,phase,batch=1,size=64):
        self.data50=CustomDataset(phase,imagesize=size)
        batch_size=batch
        super().__init__(dataset= self.data50,batch_size=batch_size)


# a=CustomDataset("./dataset/train/good")
# image = next(iter(a))
