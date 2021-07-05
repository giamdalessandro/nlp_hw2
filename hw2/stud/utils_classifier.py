import torch
import pytorch_lightning as pl


class ABSAClassifier(pl.LightningModule):
    """
    LightningModule to easly handle training and evaluation loops with a given nn.Module.
    """
    def __init__(self, input_dim=768, hidden_dim=128, output_dim=59):
        super().__init__()
        return    

    def forward(self, x, lemma, test=False):
        return

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        return

    def validation_step(self, val_batch, batch_idx):
        return

    # may be needed
    ##def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
    ##    return super().backward(loss, optimizer, optimizer_idx, retain_graph=True, *args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

