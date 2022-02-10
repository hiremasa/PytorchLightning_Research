import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from model.models import Custom_EfficientNet
from model.loss import SmoothCrossEntropyLoss

# def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
#     assert alpha > 0, "alpha should be larger than 0"
#     assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

#     lam = np.random.beta(alpha, alpha)
#     rand_index = torch.randperm(x.size()[0])
#     mixed_x = lam * x + (1 - lam) * x[rand_index, :]
#     target_a, target_b = y, y[rand_index]
#     return mixed_x, target_a, target_b, lam

class Model(pl.LightningModule):
    def __init__(self, config: dict, int_to_label=dict):
        super().__init__()
        self._config = config
        self.net = get_model(self._config)
        self._criterion = eval(self._config.loss.name)(smoothing=self._config.loss.smooth)

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(),
            torchmetrics.Precision(num_classes=self._config.num_classes, average='macro'),
            torchmetrics.Recall(num_classes=self._config.num_classes, average='macro'),
            torchmetrics.FBeta(num_classes=self._config.num_classes, beta=0.5, average='macro'),
        ])

        self.train_metrics = metrics.clone(prefix='train_')

        self.valid_metrics = metrics.clone(prefix='val_')

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        #optimizer
        if self._config.optimizer.name == 'optim.Adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self._config.optimizer.params.lr, betas=(0.9, 0.999), weight_decay=self._config.optimizer.params.weight_decay)
        elif self._config.optimizer.name == 'optim.AdamW':
            optmizer = torch.optim.AdamW(self.net.parameters(), lr=self._config.optimizer.params.lr)
        else:
            raise ValueError(f'{self._config.optimizer.name} is not supported')
        #scheduler
        if self._config.scheduler.name == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self._config.scheduler.params.T_0
                    )
        else:
            scheduler = None
            return optimizer
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        preds = logits.argmax(1)
        loss = self._criterion(logits, y)
        self.log("train_loss",
                 loss,
                 prog_bar=True,
                 logger=False,
                 on_epoch=True,
                 on_step=False,
        )

        self.train_metrics(preds, y)
        self.log_dict(self.train_metrics,
                 prog_bar=False,
                 logger=False,
                 on_epoch=True,
                 on_step=False,
                     )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        preds = logits.argmax(1)
        loss = self._criterion(logits, y)
        self.log("val_loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_epoch=True,
                 on_step=False,
        )

        self.valid_metrics(preds, y)
        self.log_dict(self.valid_metrics,
                 prog_bar=False,
                 logger=True,
                 on_epoch=True,
                 on_step=False,
                     )
        return  {"loss": loss, "preds": preds, "y": y}
