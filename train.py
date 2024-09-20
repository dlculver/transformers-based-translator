import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f"Setting device to: {DEVICE}")


# Auxiliary Training Functions

def get_lr(step_num: int, warmup_steps: int, d_model: int):
    return d_model ** -0.5 * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)

# Per epoch functions

def validate_epoch(model, validation_dl: DataLoader, epoch, n_epochs, criterion = nn.NLLLoss(), device = DEVICE):
    model.eval()
    validation_losses = []
    with torch.no_grad():
        for batch in tqdm(validation_dl):
            src_tokens = batch['src_tokens'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)

            output = model(src_tokens, tgt_input, src_mask, tgt_mask)

            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            validation_losses.append(loss.item())

    avg_val_loss = sum(validation_losses)/len(validation_losses)
    print(f"Epoch {epoch + 1}/{n_epochs}, Average validation loss: {avg_val_loss:.4f}")

def train_epoch(model, training_dl: DataLoader, optim, epoch, n_epochs, criterion = nn.NLLLoss(), device = DEVICE):
    nonlocal step_num
    model.train()
    training_losses = []

    for i, batch in tqdm(enumerate(training_dl)):
        step_num += 1
        
        # update the learning rate according to the scheduler get_lr.
        new_lr = get_lr(step_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        src_tokens = batch['src_tokens'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)

        # zero the gradients
        optimizer.zero_grad()

        output = model(src_tokens, tgt_input, src_mask, tgt_mask)

        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())

        if i % 100 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, average loss at batch {i}: {sum(epoch_loss)/len(epoch_loss):.4f}")

    print(f"Epoch {epoch + 1}/{n_epochs}, Average training loss: {sum(epoch_loss)/len(epoch_loss):.4f}")

        



        
    

# train functions

def train():
    step_num = 0
    pass

