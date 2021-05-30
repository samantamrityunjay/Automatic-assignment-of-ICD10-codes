import numpy as np 
import torch
from src.utils import calculate_metrics


def hybrid_test_results(model, hybrid_test_loader, icdtype, device):

  model.eval()
  with torch.no_grad():
    model_result = []
    targets = []
    for rnn_x, cnn_x, batch_targets in hybrid_test_loader:
      rnn_x = rnn_x.to(device)
      cnn_x = cnn_x.to(device)

      model_batch_result = model(rnn_x, cnn_x)
      model_result.extend(model_batch_result.cpu().numpy())
      targets.extend(batch_targets[icdtype].cpu().numpy())

  result = calculate_metrics(np.array(model_result), np.array(targets))
  print('-'*10 + icdtype + '-'*10)
  print(result)