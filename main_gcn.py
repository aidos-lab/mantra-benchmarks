import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from mantra.simplicial import SimplicialDataModule, SimplicialDataModuleConfig
from models.base import LitModel
from models.gcn import GCN


dm = SimplicialDataModule(SimplicialDataModuleConfig())
dm.setup()

gcnmodel = GCN(hidden_channels=64)

litmodel = LitModel(gcnmodel)

# Init trainer
trainer = Trainer(
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    accelerator="auto",
    devices=(
        1 if torch.cuda.is_available() else None
    ),  # limiting got iPython runs
    fast_dev_run=True,
)
# Pass the datamodule as arg to trainer.fit to override model hooks :)
trainer.fit(litmodel, dm)
