import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import wandb
import os

from components import EncoderDecoder
from evaluation import BLEUEvaluator


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
        train_dl: DataLoader,
        val_dl: DataLoader,
        optimizer,
        tokenizer: ByteLevelBPETokenizer,
        criterion=nn.NLLLoss(),
        device=None,
        warmup_steps: int = 5000,
        d_model: int = 512,
    ):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.device = device if device else torch.device("cpu")
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = 0

        # lists for evaluations
        self.epoch_losses = []
        self.batch_losses = []
        self.val_losses = []
        self.avg_bleu_scores = []
        self.best_val_loss = float("inf")

        # Move the model to the device and enable data parallelism if multiple GPUs are present.
        if self.device.type == "cuda":
            self.num_gpus = torch.cuda.device_count()
            print(f"Using {self.num_gpus} GPU(s)")
            if self.num_gpus > 1:
                self.model = nn.DataParallel(self.model)
        else:
            self.num_gpus = 0
            print(f"Using CPU")
        self.model.to(self.device)

        # id attributes associated to tokenizer
        self.bos = self.tokenizer.token_to_id("<s>")
        self.eos = self.tokenizer.token_to_id("</s>")
        self.pad = self.tokenizer.token_to_id("<pad>")

        # evaluators for translation
        self.bleu_evaluator = BLEUEvaluator(tokenizer, self.pad, self.bos, self.eos)

    def get_lr(self):
        """Learning Rate Scheduler as implemented in `Attention is All You Need`"""
        return self.d_model**-0.5 * min(
            self.step_num**-0.5, self.step_num * self.warmup_steps**-1.5
        )

    def train_epoch(self, epoch: int, n_epochs: int, model_save_path: str):
        self.model.train()
        training_losses = []
        device = self.device

        for i, batch in tqdm(
            enumerate(self.train_dl), total=len(self.train_dl), leave=True, position=0
        ):
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

            if i % 100 == 0 and i > 0:
                avg_loss = sum(training_losses) / len(training_losses)
                tqdm.write(
                    f"Epoch {epoch + 1}/{n_epochs}, average loss at batch {i}: {sum(training_losses)/len(training_losses):.4f}"
                )
                wandb.log({"train_loss": avg_loss, "epoch": epoch + 1, "batch": i})
            if i % 500 == 0 and i > 0:
                avg_val_loss = self._validate(epoch, n_epochs, i)
                if avg_val_loss < self.best_val_loss:
                    print(f"Validation loss has improved!")
                    self.best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), model_save_path)

                    # log model as wandb artifact
                    artifact = wandb.Artifact(f"best_model", type="model")
                    artifact.add_file(model_save_path)
                    wandb.log_artifact(artifact)

        average_training_loss = sum(training_losses) / len(training_losses)
        self.epoch_losses.append(average_training_loss)
        self.batch_losses.extend(training_losses)
        print(
            f"Epoch {epoch + 1}/{n_epochs}, Average training loss: {average_training_loss:.4f}"
        )
        wandb.log({"avg_train_loss": average_training_loss})

    def _validate(self, epoch: int, n_epochs: int, batch_num: int):
        """To be used within the training loop to get validation losses"""
        self.model.eval()
        device = self.device
        validation_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_dl, total=len(self.val_dl)):
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
            f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_num}, Average validation loss: {avg_val_loss}"
        )

        # put the model back into train mode
        self.model.train()
        return avg_val_loss

    def validate_epoch(self, epoch: int, n_epochs: int):
        self.model.eval()
        validation_losses = []
        device = self.device
        total_bleu_score = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dl):
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

                predicted_ids = torch.argmax(output, dim=-1)
                predicted_batch = [ids.tolist() for ids in predicted_ids]
                reference_batch = [ids.tolist() for ids in tgt_output]
                total_bleu_score += self.bleu_evaluator.evaluate_batch_bleu(
                    predicted_batch=predicted_batch, reference_batch=reference_batch
                )

        avg_val_loss = sum(validation_losses) / len(validation_losses)
        avg_bleu_score = total_bleu_score / len(self.val_dl)
        self.val_losses.append(avg_val_loss)
        self.avg_bleu_scores.append(avg_bleu_score)
        print(
            f"Epoch: {epoch + 1}/{n_epochs}, Average validation loss: {avg_val_loss: .4f}"
        )
        wandb.log({"val_loss": avg_val_loss, "avg_bleu_score": avg_bleu_score})

        return avg_val_loss

    def train(
        self,
        n_epochs: int,
        save_dir: str,
    ):
        """Main function to handle the full training and validation process."""
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, "model.pth")
        loss_save_path = os.path.join(save_dir, "losses.pth")
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            self.train_epoch(
                epoch=epoch, n_epochs=n_epochs, model_save_path=model_save_path
            )
            avg_val_loss = self.validate_epoch(epoch=epoch, n_epochs=n_epochs)

            if avg_val_loss < self.best_val_loss:
                print(f"Validation loss has improved!")
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), model_save_path)

                # log model as wandb artifact
                artifact = wandb.Artifact(f"best_model", type="model")
                artifact.add_file(model_save_path)
                wandb.log_artifact(artifact)

        print("Training complete...")
        print(f"Best validation loss: {self.best_val_loss: .4f}")
        print(f"Saving losses for later use...")
        torch.save(
            {
                "training_losses": self.epoch_losses,
                "per_batch_losses": self.batch_losses,
                "validation_losses": self.val_losses,
                "bleu_scores": self.avg_bleu_scores,
            },
            loss_save_path,
        )
        wandb.save(loss_save_path)


if __name__ == "__main__":

    from data_processing import TranslationDataset, collate_fn
    from datasets import load_dataset
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train a transformer model for translation."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=3, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="test_model",
        help="Directory to save the model and losses.",
    )
    parser.add_argument(
        "--tokenizer_vocab",
        type=str,
        default="transformers-based-translator/de-en-bpetokenizer/vocab.json",
        help="Path to the tokenizer vocab file.",
    )
    parser.add_argument(
        "--tokenizer_merges",
        type=str,
        default="transformers-based-translator/de-en-bpetokenizer/merges.txt",
        help="Path to the tokenizer merges file.",
    )
    args = parser.parse_args()

    import wandb

    tokenizer = ByteLevelBPETokenizer(
        "bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt"
    )

    PAD_TOKEN_ID = tokenizer.token_to_id("<pad>")
    BOS_TOKEN_ID = tokenizer.token_to_id("<s>")
    EOS_TOKEN_ID = tokenizer.token_to_id("</s>")
    src_lang = "de"
    tgt_lang = "en"

    config = {
        "num_blocks": 6,
        "num_heads": 8,
        "d_model": 512,
        "d_ff": 2048,
        "vocab_size": tokenizer.get_vocab_size(),
        "max_len": 512,
        "dropout": 0.1,
        "verbose": False,
        "betas": (0.9, 0.98),
        "eps": 1e-9,
        "warmup_steps": 5000,
        "batch_size": 64,
        "num_epochs": 3,
    }

    api_key = os.getenv("WANDB_API_KEY")

    wandb.init(project="transformers-based-translator", config=config)

    # Download and create Translation Datasets.
    dataset = load_dataset("wmt14", "de-en")
    train_ds = TranslationDataset(
        dataset["train"].shuffle(),
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )
    val_ds = TranslationDataset(
        dataset["validation"],
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )

    # create dataloaders
    batch_size = config["batch_size"]
    num_workers = 16
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, PAD_TOKEN_ID),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, PAD_TOKEN_ID),
        num_workers=num_workers,
        pin_memory=True,
    )

    # Instantiate a model
    model = EncoderDecoder.from_hyperparameters(
        **config
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    trainer = Trainer(
    model=model,
    train_dl=train_dl,
    val_dl=val_dl,
    tokenizer=tokenizer,
    optimizer=optimizer,
    criterion=torch.nn.NLLLoss(),
    device=DEVICE,
)

    trainer.train(n_epochs=5, save_dir="test_model")
