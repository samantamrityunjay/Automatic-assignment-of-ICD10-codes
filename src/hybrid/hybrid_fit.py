
import torch
from src.utils import train_metric



def hybrid_fit(epochs, model, hybrid_train_loader, hybrid_val_loader, icdtype, opt_fn,loss_fn, learning_rate, device):
  optimizer = opt_fn(model.parameters(), lr=learning_rate)
  print('-'*10 + icdtype + '-'*10)
  for epoch in range(1,epochs+1):

    model.train()

    train_epoch_loss=0
    train_epoch_accuracy=0
    train_epoch_hammingloss=0
    train_epoch_f1score=0

    val_epoch_loss=0
    val_epoch_accuracy=0
    val_epoch_hammingloss=0
    val_epoch_f1score=0
    
    for rnn_x, cnn_x, y_dict in hybrid_train_loader:

      rnn_x = rnn_x.to(device)
      cnn_x = cnn_x.to(device)

      y = y_dict[icdtype]
      y = y.to(device)
      

      
      preds=model(rnn_x, cnn_x)

      optimizer.zero_grad()
      loss=loss_fn(preds,y)
      loss.backward()
      optimizer.step()
      
      accuracy, hammingloss, f1score  = train_metric(preds,y)

      train_epoch_loss+=loss.item()
      train_epoch_accuracy+=accuracy.item()
      train_epoch_hammingloss+=hammingloss
      train_epoch_f1score+=f1score
    
    model.eval()
    with torch.no_grad():
      for rnn_x, cnn_x, y_dict in hybrid_val_loader:
        
        rnn_x = rnn_x.to(device)
        cnn_x = cnn_x.to(device)

        y = y_dict[icdtype]
        y = y.to(device)
        
        preds=model(rnn_x, cnn_x)

        loss=loss_fn(preds,y)
        accuracy, hammingloss, f1score  = train_metric(preds,y)
        val_epoch_loss+=loss.item()
        val_epoch_accuracy+=accuracy.item()
        val_epoch_hammingloss+=hammingloss
        val_epoch_f1score+=f1score

    
    
    print("\n")
    print('-'*100)
    print('Epoch = {}/{}:\n train_loss = {:.4f}, train_accuracy = {:.4f}, train_hammingloss = {:.4f}, train_f1score = {:.4f}\n val_loss = {:.4f}, val_accuracy = {:.4f}, val_hammmingloss = {:.4f}, val_f1score = {:.4f}'.format(epoch
                                                              ,epochs
                                                              ,train_epoch_loss/len(hybrid_train_loader)
                                                              ,train_epoch_accuracy/len(hybrid_train_loader)
                                                              ,train_epoch_hammingloss/len(hybrid_train_loader)
                                                              ,train_epoch_f1score/len(hybrid_train_loader)
                                                              ,val_epoch_loss/len(hybrid_val_loader)
                                                              ,val_epoch_accuracy/len(hybrid_val_loader)
                                                              ,val_epoch_hammingloss/len(hybrid_val_loader)
                                                              ,val_epoch_f1score/len(hybrid_val_loader)
                                                              ))
    print('-'*100)
    print("\n")
    