import torch
import torch.nn as nn
import torch.optim as optim


class StockPriceGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(StockPriceGRU, self).__init__()
        
        # Define GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        # Define fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward pass through GRU layer
        out, _ = self.gru(x)
        
        # Get the output from the last time step
        out = out[:, -1, :]  # Select the output from the last time step
        
        # Pass through the output layer
        out = self.fc(out)
        return out