import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

from data import FaceLandmarkDataset, ToTensor, add_padding_to_make_square


class MobileNetV3Landmark(nn.Module):
    def __init__(self, num_landmarks=136):
        super(MobileNetV3Landmark, self).__init__()
        self.mobilenet = mobilenet_v3_large(pretrained=True)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_landmarks)

    def forward(self, x):
        x = self.mobilenet(x)
        return x


def train_model(model, train_loader, valid_loader, num_epochs=25, learning_rate=0.001, early_stopping_patience=5):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = nn.DataParallel(model, device_ids=[0, 1])
    else:
        device = torch.device('cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    no_improve_epoch = 0

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            inputs, labels = data['image'].to(device), data['landmark'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 136))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
	
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data['image'].to(device), data['landmark'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 136))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valid_loader)
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Valid Loss: {avg_val_loss}\n')
        
        # earlystopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= early_stopping_patience:
                print('Early stopping triggered.')
                break
        model.train()

    print('Finished Training')



if __name__=="__main__":
    transform = transforms.Compose([
        transforms.Lambda(add_padding_to_make_square),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
        ])
    
    user_metadata = pd.read_csv('/home/ysj/face_landmark/user_metadata_split.csv')
    train_user = user_metadata[user_metadata['split_data']=='train']
    valid_user = user_metadata[user_metadata['split_data']=='valid']
    test_user = user_metadata[user_metadata['split_data']=='test']

    train_dataset = FaceLandmarkDataset(
        root_dir='/home/ysj/face_landmark/dataset/',
        user_metadata=train_user['user_metadata'].values,
	transform=transform
        )
    valid_dataset = FaceLandmarkDataset(
        root_dir='/home/ysj/face_landmark/dataset/',
        user_metadata=valid_user['user_metadata'].values,
        transform=transform
        )
    test_dataset = FaceLandmarkDataset(
        root_dir='/home/ysj/face_landmark/dataset/',
        user_metadata=test_user['user_metadata'].values,
        transform=transform
        )
    print(f'#### train_dataset:{len(train_dataset)} / valid_dataset:{len(valid_dataset)} / test_dataset:{len(test_dataset)} ####')
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=4)
    #test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)
   
    
    model = MobileNetV3Landmark()
    train_model(model, train_loader, valid_loader, num_epochs=25, learning_rate=0.001, early_stopping_patience=5)


 
