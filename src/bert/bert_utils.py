import numpy as np 
import torch
from src.utils import calculate_metrics

#################### Printing Test Metrics ##############################
def bert_test_results(model, test_loader, icdtype, device):

  model.eval()
  with torch.no_grad():
    model_result = []
    targets = []
    for resps, batch_targets in test_loader:
      model_batch_result = model(resps["ids"].to(device), resps["mask"].to(device), resps["token_type_ids"].to(device))
      model_result.extend(model_batch_result.cpu().numpy())
      targets.extend(batch_targets[icdtype].cpu().numpy())
  result = calculate_metrics(np.array(model_result), np.array(targets))
  print('-'*20 + icdtype + '-'*20)
  print(result)
  print('-'*len('-'*20 + icdtype + '-'*20))

########################################################################