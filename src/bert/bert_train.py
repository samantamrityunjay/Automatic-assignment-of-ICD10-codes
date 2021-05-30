import torch
from src.utils import train_metric

def bert_fit(epochs, model, train_loader, val_loader, icdtype, opt_fn, loss_fn, lr, device):

    optimizer = opt_fn(model.parameters(), lr=lr)
    print("-" * 10 + icdtype + "-" * 10)
    for epoch in range(1, epochs + 1):

        model.train()

        train_epoch_loss = 0
        train_epoch_accuracy = 0
        train_epoch_hammingloss = 0
        train_epoch_f1score = 0

        val_epoch_loss = 0
        val_epoch_accuracy = 0
        val_epoch_hammingloss = 0
        val_epoch_f1score = 0

        for resps, targets in train_loader:

            target = targets[icdtype]
            target = target.to(device)

            preds = model(
                resps["ids"].to(device),
                resps["mask"].to(device),
                resps["token_type_ids"].to(device),
            )

            optimizer.zero_grad()
            loss = loss_fn(preds, target)
            loss.backward()
            optimizer.step()

            accuracy, hammingloss, f1score = train_metric(preds, target)

            train_epoch_loss += loss.item()
            train_epoch_accuracy += accuracy.item()
            train_epoch_hammingloss += hammingloss
            train_epoch_f1score += f1score

        model.eval()
        with torch.no_grad():
            for resps, targets in val_loader:

                target = targets[icdtype]
                target = target.to(device)

                preds = model(
                    resps["ids"].to(device),
                    resps["mask"].to(device),
                    resps["token_type_ids"].to(device),
                )

                loss = loss_fn(preds, target)
                accuracy, hammingloss, f1score = train_metric(preds, target)

                val_epoch_loss += loss.item()
                val_epoch_accuracy += accuracy.item()
                val_epoch_hammingloss += hammingloss
                val_epoch_f1score += f1score

        print("\n")
        print("-" * 100)
        print(
            "Epoch = {}/{}:\n train_loss = {:.4f}, train_accuracy = {:.4f}, train_hammingloss = {:.4f}, train_f1score = {:.4f}\n val_loss = {:.4f}, val_accuracy = {:.4f}, val_hammmingloss = {:.4f}, val_f1score = {:.4f}".format(
                epoch,
                epochs,
                train_epoch_loss / len(train_loader),
                train_epoch_accuracy / len(train_loader),
                train_epoch_hammingloss / len(train_loader),
                train_epoch_f1score / len(train_loader),
                val_epoch_loss / len(val_loader),
                val_epoch_accuracy / len(val_loader),
                val_epoch_hammingloss / len(val_loader),
                val_epoch_f1score / len(val_loader),
            )
        )
        print("-" * 100)
        print("\n")
