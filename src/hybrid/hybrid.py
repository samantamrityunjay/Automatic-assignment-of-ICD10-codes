import torch 
import torch.nn as nn
from src.rnn.rnn_utils import create_emb_layer


class hybrid(nn.Module):
  def __init__(self, vocabulary, sequence_length, weights_matrix, hidden_size, num_layers=2, num_classes=10):
    super().__init__()

    self.num_layers = num_layers
    self.hidden_size = hidden_size

    self.conv1 = nn.Sequential(nn.Conv1d(len(vocabulary)+1,
                                            128,
                                            kernel_size=7,
                                            padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool1d(3)
                                  )

    self.conv2 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=7, padding=0),
                                nn.ReLU(),
                                nn.MaxPool1d(3)
                                )

    self.conv3 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, padding=0),
                                nn.ReLU()
                                )

    self.conv4 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, padding=0),
                                nn.ReLU()
                                )
    input_shape = (1, len(vocabulary)+1, sequence_length)
    self.output_dimension = self._get_conv_output(input_shape)

    # define linear layers

    self.fc1 = nn.Sequential(
        nn.Linear(self.output_dimension, 256),
        nn.ReLU(),
    )

      
    self.embeddings, num_embeddings, embedding_size = create_emb_layer(weights_matrix, True)
    self.gru1 = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional = True, batch_first=True)
        
    self.fc2 = nn.Sequential(
        nn.Linear(2*hidden_size, 256),
        nn.ReLU(),
    ) 


    self.fc3 = nn.Linear(512,10)   
    self.act = nn.Sigmoid()


  def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension
  
  def forward(self,rnninput, cnninput):
    cnn_out = self.conv1(cnninput)
    cnn_out = self.conv2(cnn_out)
    cnn_out = self.conv3(cnn_out)
    cnn_out = self.conv4(cnn_out)
    cnn_out = cnn_out.view(cnn_out.size(0),-1)
    cnn_out = self.fc1(cnn_out)

    rnn_out = self.embeddings(rnninput)
    rnn_out,_ = self.gru1(rnn_out)
    rnn_out = self.fc2(rnn_out[:,-1,:])


    x = torch.cat((cnn_out,rnn_out),dim=1)
    out = self.fc3(x)
    out = self.act(out)
    return out
