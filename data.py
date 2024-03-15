import os
import json
import numpy as np
import pandas as pd

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


class FaceLandmarkDataset(Dataset):
    def __init__(self, root_dir, user_metadata, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_file = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if 'image' not in root: continue
                user = root.split('/')[-2]
                if user in user_metadata:
                    self.image_file.append(os.path.join(root, file))        

    def __len__(self):
        return len(self.image_file)

    def __getitem__(self, idx):
        img_name = self.image_file[idx]
        json_name = img_name.replace('image', 'json').replace('.jpg', '.json')

        image = Image.open(img_name).convert('RGB')
        with open(json_name) as f:
            landmark_data = json.load(f)
        landmark = landmark_data['landmark']
        landmark = np.array(landmark)
        
        data = {'image': image, 'landmark':landmark}
        if self.transform:
            data = self.transform(data)
 
        return data



class ResizeTransform:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        new_image = F.resize(image, self.output_size)
        return new_image



class RescaleAndAdjustLandmark(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
   
    def __call__(self, data):
        image, landmark = data['image'], data['landmark']
        
        # image rescale
        padding= (max(image.size) - min(image.size)) // 2
        new_size = [max(image.size), max(image.size)]
        new_img = Image.new('RGB', new_size)
        new_img.paste(image, (0, padding))
        new_img = new_img.resize((self.output_size, self.output_size))
        
        # Adjust landmark
        landmark[:, 1] += padding
        landmark = landmark.astype(float)
        landmark *= self.output_size / max(image.size)

        return {'image': new_img, 'landmark': landmark.flatten()} 



class ToTensor(object):
    def __call__(self, data):
        image, landmark = data['image'], data['landmark']
        image = transforms.ToTensor()(image).float()
        landmark = torch.from_numpy(landmark).float()
        return {'image':image, 'landmark':landmark}



class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data):
        image, landmark = data['image'], data['landmark']
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return {'image':image, 'landmark':landmark}



def add_padding_to_make_square(image):
    original_width, original_height = image.size
    target_size = max(original_width, original_height)
    top = bottom = (target_size - original_height) // 2
    left = right = (target_size - original_width) // 2
    new_image = F.pad(image, (left, top, right, bottom), fill=0)
    return new_image



if __name__=="__main__":
    transform = transforms.Compose([
        RescaleAndAdjustLandmark(224),
        ToTensor(),
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
        ])

    user_metadata = pd.read_csv('/home/ysj/face_landmark/user_metadata_split.csv')
    train = user_metadata[user_metadata['split_data']=='train']
    transformed_dataset = FaceLandmarkDataset(
            root_dir='/home/ysj/face_landmark/dataset/',
            user_metadata=train['user_metadata'].values,
            transform=transform
            )
    print(f'### {len(transformed_dataset)} ###')
    data_loader = DataLoader(transformed_dataset, batch_size=32, shuffle=True, num_workers=4)
   
    data = transformed_dataset[0]
    print(len(data))
    print(data)
