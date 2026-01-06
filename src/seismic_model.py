import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import F1Score, HammingDistance

class SeismicMultilabelModel(pl.LightningModule):
    """
    PyTorch Lightning model for multilabel seismic event classification.

    Args:
        input_dim (int): Number of input features.
        hidden_units (tuple): Sizes of hidden layers.
        dropout_rate (float): Dropout probability between layers.
        lr (float): Learning rate for the optimizer.
        l2 (float): Weight decay (L2 regularization).
        num_classes (int): Number of output labels.
        **kwargs: Additional hyperparameters (ignored).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units=(128,),
        dropout_rate: float = 0.0,
        lr: float = 1e-3,
        l2: float = 1e-4,
        num_classes: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build MLP with a Flatten at the front
        layers = [nn.Flatten()]
        prev_dim = self.hparams.input_dim
        for h in self.hparams.hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if self.hparams.dropout_rate > 0:
                layers.append(nn.Dropout(self.hparams.dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, self.hparams.num_classes))

        # Keep both names so .ckpt loading via "classifier.*" works
        self.model      = nn.Sequential(*layers)
        self.classifier = self.model

        # Loss + metrics
        self.criterion      = nn.BCEWithLogitsLoss()
        self.train_f1       = F1Score(task="multilabel", num_labels=self.hparams.num_classes, average="micro")
        self.train_hamming  = HammingDistance(task="multilabel", num_labels=self.hparams.num_classes)
        self.val_f1         = F1Score(task="multilabel", num_labels=self.hparams.num_classes, average="micro")
        self.val_hamming    = HammingDistance(task="multilabel", num_labels=self.hparams.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss   = self.criterion(logits, y.float())

        probs = torch.sigmoid(logits)
        self.train_f1.update(probs, y)
        self.train_hamming.update(probs, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_f1_micro",  self.train_f1.compute(),    prog_bar=True)
        self.log("train_hamming",   self.train_hamming.compute(), prog_bar=True)
        self.train_f1.reset()
        self.train_hamming.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss   = self.criterion(logits, y.float())

        probs = torch.sigmoid(logits)
        self.val_f1.update(probs, y)
        self.val_hamming.update(probs, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_f1_micro",  self.val_f1.compute(),    prog_bar=True)
        self.log("val_hamming",   self.val_hamming.compute(), prog_bar=True)
        self.val_f1.reset()
        self.val_hamming.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss   = self.criterion(logits, y.float())
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.l2
        )