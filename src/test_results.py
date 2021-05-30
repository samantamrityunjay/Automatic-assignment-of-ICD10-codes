import numpy as np 
import torch
from src.utils import calculate_metrics

def test_results(model, test_loader, icdtype, device):

  model.eval()
  with torch.no_grad():
    model_result = []
    targets = []
    for x_test, batch_targets in test_loader:
      x_test = x_test.to(device)
      model_batch_result = model(x_test)
      model_result.extend(model_batch_result.cpu().numpy())
      targets.extend(batch_targets[icdtype].cpu().numpy())
  result = calculate_metrics(np.array(model_result), np.array(targets))
  print('-'*10 + icdtype + '-'*10)
  print(result)