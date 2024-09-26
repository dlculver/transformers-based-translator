import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import os

from components import EncoderDecoder


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Setting device to: {DEVICE}")


class Trainer:
    def __init__(
        self,
        model: EncoderDecoder,
        optimizer,
        criterion=nn.NLLLoss(),
        device=DEVICE,
        warmup_steps: int = 5000,
        d_model: int = 512,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = 0

        self.epoch_losses = []
        self.batch_losses = []
        self.val_losses = []

        self.best_val_loss = float('inf')

        self.model.to(self.device)

    def get_lr(self):
        """Learning Rate Scheduler as implemented in `Attention is All You Need`"""
        return self.d_model**-0.5 * min(
            self.step_num**-0.5, self.step_num * self.warmup_steps**-1.5
        )

    def train_epoch(self, training_dl: DataLoader, epoch: int, n_epochs: int):
        self.model.train()
        training_losses = []
        device = self.device

        for i, batch in tqdm(enumerate(training_dl)):
            self.step_num += 1

        # update the learning rate according to the scheduler get_lr.
        new_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        src_tokens = batch["src_tokens"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        # zero the gradients
        self.optimizer.zero_grad()

        output = self.model(src_tokens, tgt_input, src_mask, tgt_mask)

        loss = self.criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        loss.backward()
        self.optimizer.step()

        training_losses.append(loss.item())

        if i % 100 == 0:
            tqdm.write(
                f"Epoch {epoch + 1}/{n_epochs}, average loss at batch {i}: {sum(training_losses)/len(training_losses):.4f}"
            )

        average_training_loss = sum(training_losses)/len(training_losses)
        self.epoch_losses.append(average_training_loss)
        self.batch_losses.extend(training_losses)
        print(
            f"Epoch {epoch + 1}/{n_epochs}, Average training loss: {average_training_loss:.4f}"
        )

    def validate_epoch(self, validation_dl: DataLoader, epoch: int, n_epochs: int):
        self.model.eval()
        validation_losses = []
        device = self.device

        with torch.no_grad():
            for batch in tqdm(validation_dl):
                src_tokens = batch["src_tokens"].to(device)
                tgt_input = batch["tgt_input"].to(device)
                tgt_output = batch["tgt_output"].to(device)
                src_mask = batch["src_mask"].to(device)
                tgt_mask = batch["tgt_mask"].to(device)

                output = self.model(src_tokens, tgt_input, src_mask, tgt_mask)

                loss = self.criterion(
                    output.view(-1, output.size(-1)), tgt_output.view(-1)
                )
                validation_losses.append(loss.item())

        avg_val_loss = sum(validation_losses) / len(validation_losses)
        self.val_losses.append(avg_val_loss)
        print(
            f"Epoch: {epoch + 1}/{n_epochs}, Average validation loss: {avg_val_loss: .4f}"
        )

        return avg_val_loss

    def train(
        self,
        training_dl: DataLoader,
        validation_dl: DataLoader,
        n_epochs: int,
        save_dir: str,
    ):
        """Main function to handle the full training and validation process."""
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, "model.pth")
        loss_save_path = os.path.join(save_dir, "losses.pth")
        for epoch in tqdm(range(n_epochs)):
            self.train_epoch(training_dl=training_dl, epoch=epoch, n_epochs=n_epochs)
            avg_val_loss = self.validate_epoch(
                validation_dl=validation_dl, epoch=epoch, n_epochs=n_epochs
            )

            if avg_val_loss < self.best_val_loss:
                print(f"Validation loss has improved!")
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), model_save_path)

        print("Training complete...")
        print(f"Best validation loss: {self.best_val_loss: .4f}")
        print(f"Saving losses for later use...")
        torch.save({
            'training_losses': self.epoch_losses,
            'per_batch_losses': self.batch_losses,
            'validation_losses': self.val_losses
        }, loss_save_path)


if __name__ == "__main__":

    from data_processing import TranslationDataset, collate_fn
    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer(
        "bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt"
    )

    PAD_TOKEN_ID = tokenizer.token_to_id("<pad>")
    BOS_TOKEN_ID = tokenizer.token_to_id("<s>")
    EOS_TOKEN_ID = tokenizer.token_to_id("</s>")
    src_lang = "de"
    tgt_lang = "en"

    dataset = load_dataset("wmt14", "de-en")
    train_ds = TranslationDataset(
        dataset["train"].shuffle().select(range(100)),
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )
    val_ds = TranslationDataset(
        dataset["validation"].select(range(10)),
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )

    model = EncoderDecoder.from_hyperparameters(
        num_blocks=6,
        num_heads=8,
        d_model=512,
        d_ff=2048,
        vocab_size=tokenizer.get_vocab_size(),
    )

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        ),
        criterion=nn.NLLLoss(),
        device=DEVICE,
        warmup_steps=5000,
        d_model=512,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, PAD_TOKEN_ID),
    )
    valid_dl = DataLoader(
        val_ds, batch_size=4, collate_fn=lambda batch: collate_fn(batch, PAD_TOKEN_ID)
    )

    trainer.train(train_dl, valid_dl, n_epochs=4, save_dir="test_model")
