import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import fwc2
from FWC2L.fwc2.datasets import DataPreperationPipeline
from torch.utils.data import DataLoader

def train_model(model, train_loader, optimizer, criterion, corruption=None, device='cpu', epochs=100):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x, _ = x.to(device)
            
            corrupted = corruption.corrupt(x)
            
            _, proj1 = model(x)
            _, proj2 = model(corrupted)
            
            loss = criterion(proj1, proj2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dim = 64
    batch_size = 128
    learning_rate = 0.001
    epochs = 2
    courpution_ratio = 0.2
    tau = 0.5
    
    pipline = DataPreperationPipeline(base_path='./datasets', dataset_name='dapt20')
    pipline.run()

    train_loader = DataLoader(pipline.training_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(pipline.testing_dataset, batch_size=batch_size, shuffle=True)
        
    model = fwc2.FWC2Model(input_dim=input_dim).to(device)
    
    corruption = fwc2.FeatureCorruption(X=pipline.training_dataset, corruption_ratio=courpution_ratio)
    criterion = fwc2.NTXentLoss(temperature=tau).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, optimizer, criterion, corruption, device, epochs)
    
if __name__ == "__main__":
    main()