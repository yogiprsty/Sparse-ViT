import torch
from tqdm import tqdm
import utils

def train_per_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epochs, epoch, device):
    epoch_acc = 0
    epoch_loss = 0
    n_train_steps = len(train_loader)
    
    with tqdm(train_loader, unit="batch") as tepoch:
      for (images, labels) in tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs, sparse = model(images)
        running_loss = criterion(outputs, labels)
        running_acc = utils.calculate_accuracy(outputs, labels)
        
        running_loss += 0.01 * sparse
        
        running_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        epoch_acc += running_acc / n_train_steps
        epoch_loss += running_loss / n_train_steps

        tepoch.set_postfix(loss=running_loss.item(), accuracy=running_acc.item())
    
    return (
        (epoch_loss.item(), epoch_acc.item())
    )

def train(model, checkpoint_path, train_loader, criterion, optimizer, lr_scheduler, epochs, device, history_path):    
    best_loss = 1000
    model.train()

    if not os.path.exists(history_path):
        os.makedirs(history_path)

    with open(history_path, 'a') as history:
        history.write('epoch,loss,accuracy\n')
    
    for epoch in range(0, epochs):
        model.train(True)
        (epoch_loss, epoch_acc) = train_per_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epochs, epoch, device)
        
        # save progress
        with open(history_path, 'a') as history:
            history.write('{},{},{}\n'.format(epoch, epoch_loss, epoch_acc))
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
            # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict()
            }, checkpoint_path)
            
            print('model saved to {}, loss: {:.4f}'.format(checkpoint_path, epoch_loss))
    print('==== Training Finished ====')