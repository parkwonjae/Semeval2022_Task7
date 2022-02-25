from datetime import datetime
from typing import Optional

import datasets
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import os

from data import GLUEDataModule, SemEvalDataModule
from model import GLUETransformer, SemEvalTransformer
from setup import AVAIL_GPUS


from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(name='electra base', project='pytorchlightning')

seed_everything(0)

# # setup data
# dm = GLUEDataModule(model_name_or_path="albert-base-v2", task_name="mnli")
# dm.setup("fit")
#
# # setup model
# model = GLUETransformer(
#     model_name_or_path="albert-base-v2",
#     num_labels=dm.num_labels,
#     eval_splits=dm.eval_splits,
#     task_name=dm.task_name,
# )
#
# # define a trainer
# trainer = Trainer(max_epochs=3, gpus=AVAIL_GPUS, logger=wandb_logger)
# trainer.validate(model, dm.val_dataloader())

##################

os.environ["TOKENIZERS_PARALLELISM"] = "true"
dm = SemEvalDataModule(model_name_or_path='google/electra-base-discriminator')
#dm.setup('fit')

model = SemEvalTransformer(
    model_name_or_path='google/electra-base-discriminator',
    num_labels=dm.num_labels,
)


trainer = Trainer(
    max_epochs=20,
    gpus=AVAIL_GPUS,
    logger=wandb_logger,
    profiler="simple"
    # progress_bar_refresh_rate=20
    # 이렇게 하면 20 ... 40 ... 이런식으로 fresh만됨.
)
trainer.fit(
    model,
    datamodule=dm
)