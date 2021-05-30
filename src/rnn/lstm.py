import torch 
import torch.nn as nn
from src.rnn.rnn_utils import create_emb_layer

class LSTMw2vmodel(nn.Module) :
  def __init__(self, weights_matrix, hidden_size, num_layers, device, num_classes = 10) :

    super().__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.device = device
    self.embeddings, num_embeddings, embedding_size = create_emb_layer(weights_matrix, True)
    self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional = True)
    self.fc1 = nn.Sequential(
        nn.Linear(2*hidden_size, 128),
        nn.ReLU(),
    )
    self.fc2 = nn.Linear(128, num_classes)
    self.act = nn.Sigmoid()
      
      
  def forward(self, x):     
    x = self.embeddings(x)
    h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device)
    c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device)
    lstm_out, (ht, ct) = self.lstm(x, (h0,c0))

    out = self.fc1(lstm_out[:,-1,:])
    out = self.fc2(out)
    return self.act(out)