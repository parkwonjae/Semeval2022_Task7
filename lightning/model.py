import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AdamW, AutoTokenizer, AutoConfig, PreTrainedTokenizer, PretrainedConfig
from pathlib import Path
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers import AutoModelForSequenceClassification
from pytorch_lightning.loggers import WandbLogger
from overrides import overrides

wandb_logger = WandbLogger(name='test', project='pytorchlightning')

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

class BaseModel(pl.LightningModule):
    USE_TOKEN_TYPE_MODELS = ["bert", "electra"]

    def __init__(
            self,
            args: argparse.Namespace,
            num_lables: Optional[int] = None,
            config: Optional[PretrainedConfig] = None,
            model_type: Optional[str] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            metrics: Dict[str, Any] = {},
            **config_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        data = getattr(args, "data", None)
        if data is not None:
            delattr(args, "data")
        self.save_hyperparameters(args)
        self.args.data = data
        self.step_count = 0
        self.output_dir = Path(self.args.output_dir)
        self.predictions: List[int] = []

        cache_dir = self.args.cache_dir if self.args.cache_dir else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.args.config_name if self.args.config_name else self.args.model_name_or_path,
                **({"num_labels": num_lables} if num_lables is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name if self.args.tokenizer_name else self.args.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer = tokenizer
        self.model = model_type.from_pretrained(
            self.args.model_name_or_path,
            from_tf=bool(".ckpt" in self.args.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )
        self.metrics = nn.ModuleDict(metrics)
        self.dev_dataset_type = 'dev'
        self.test_dataset_type = 'test'

    def get_lr_scheduler(self) -> Any:
        get_schedule_func = arg_to_scheduler[self.args.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.num_warmup_steps(), num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n,p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.args.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.args.learning_rate, scale_parameter=False, relative_step=False
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon
            )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]

    def training_step(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_step_end(self, training_step_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"loss": training_step_outputs["loss"].mean()}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int, data_type:str) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def validation_epoch_end(self,
         outputs: List[Dict[str, torch.Tensor]], data_type: str='dev', write_predictions: bool = False
    ) -> None:
        preds = self._convert_outputs_to_preds(outputs)
        labels = torch.cat([output["labels"] for output in outputs], dim=0)
        if write_predictions is True:
            self.predictions = preds
        self._set_metrics_device()
        for k, metric in self.metrics.items():
            metric(preds, labels)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        assert self.dev_dataset_type in {"dev", "test"}
        return self.validation_step(batch, batch_idx, data_type=self.dev_dataset_type)

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        assert self.dev_dataset_type in {"dev", "test"}
        return self.validation_epoch_end(outputs, data_type=self.dev_dataset_type, write_predictions=True)

    def _convert_outputs_to_preds(self, outputs: List[Dict[str, torch.Tensor]]) -> Any:
        return NotImplementedError

    def _set_metrics_device(self) -> None:
        device = next(self.parameters()).device
        for _, metric in self.metrics.items():
            if metric.device is None:
                metric.device = device

    def num_warmup_steps(self) -> Any:
        num_warmup_steps = self.args.warmup_steps
        if num_warmup_steps is None and self.args.warmup_ratio is not None:
            num_warmup_steps = self.total_steps() * self.args.warmup_ratio
            num_warmup_steps = math.ceil(num_warmup_steps)

        if num_warmup_steps is None:
            num_warmup_steps = 0
        return num_warmup_steps

    def total_steps(self) -> Any:
        num_devices = max(1, self.args.num_gpus)
        effective_batch_size = self.args.train_batch_size * self.args.accumulate_grad_batches * num_devices
        return (self.args.dataset_size / effective_batch_size)

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("basemodel")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        #parser.add_argument('--model_name_or_path', default=None, type=str)
        parser.add_argument('--config_name', default="", type=str)
        parser.add_argument('--tokenizer_name', default=None, type=str)
        parser.add_argument('--cache_dir', default="", type=str)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--learning_rate', default=5e-5, type=float)
        parser.add_argument(
            '--lr_scheduler',
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=None, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--warmup_ratio", default=None, type=float, help="Linear warmup over warmup_step ratio.")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=4, type=int)
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument("--verbose_step_count", default=100, type=int)
        return parser


class SinglePlausibleClassification(BaseModel):
    def __init__(self, args: Union[Dict[str, Any], argparse.Namespace], metrics: Dict[str, Any] = {}) -> None:
        if type(args) == dict:
            args = argparse.Namespace(**args)
        super().__init__(
            args,
            num_labels=args.num_labels,
            model_type=AutoModelForSequenceClassification,
        )

    def forward(self, **inputs: torch.Tensor) -> Any:
        return self.model(**inputs)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.is_use_token_type():
            inputs["token_type_ids"] = batch[2]

        outputs = self(**inputs)
        loss = outputs[0]

        self.log("train/loss", loss)
        return {"loss": loss}

    def validation_step(
            self, batch: List[torch.Tensor], batch_idx: int, data_type: str = "valid"
    ) -> Dict[str, torch.Tensor]:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.is_use_token_type():
            inputs["token_type_ids"] = batch[2]

        outputs = self(**inputs)
        loss, logits = outputs[:2]

        self.log(f"{data_type}/loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"logits": logits, "labels": inputs["labels"]}

    def validation_epoch_end(
            self, outputs: List[Dict[str, torch.Tensor]], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        labels = torch.cat([output["labels"] for output in outputs], dim=0)
        preds = self._convert_outputs_to_preds(outputs)

        if write_predictions is True:
            self.predictions = preds

        self._set_metrics_device()
        for k, metric in self.metrics.items():
            metric(preds, labels)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    def _convert_outputs_to_preds(self, outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        # logits: (B, num_labels)
        logits = torch.cat([output["logits"] for output in outputs], dim=0)
        return torch.argmax(logits, dim=1)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        BaseModel.add_specific_args(parser)
        return parser