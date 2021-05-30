import torch 
import torch.nn as nn
from src.rnn.rnn_utils import create_emb_layer

class GRUw2vmodel(nn.Module) :
  def __init__(self, weights_matrix, hidden_size, num_layers, device, num_classes = 10) :

    super().__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.device = device
    self.embeddings, num_embeddings, embedding_size = create_emb_layer(weights_matrix, True)
    self.gru1 = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        
    self.fc1 = nn.Linear(hidden_size, 10)
        
    self.act = nn.Sigmoid()
      
      
  def forward(self, x):     
    x = self.embeddings(x)
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
    gru_out, _ = self.gru1(x, h0)
    out = self.fc1(gru_out[:,-1,:])
    return self.act(out)