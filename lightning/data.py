import argparse

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from typing import Optional, Dict, List, Union
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
from lightning.utils import convert_examples_to_features
from lightning.inputdataclass import InputFeatures, InputExample
from overrides import overrides

wandb_logger = WandbLogger(name='test', project='pytorchlightning')





class DataProcessor:
    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer) -> None:
        self.args = args
        self.tokenizer = tokenizer

    def get_train_dataset(self, data_dir: str, data_file: str, label_file: str) -> Dataset:
        data_file_path = os.path.join(data_dir, 'train', data_file)
        label_file_path = os.path.join(data_dir, 'train', label_file)
        wandb_logger.log_text("Loading from {}, {}".format(data_file_path, label_file_path))
        return self._create_dataset(
            data_file_path=data_file_path,
            label_file_path=label_file_path,
            dataset_type="train"
        )

    def get_dev_dataset(self, data_dir: str, data_file: str, label_file: str) -> Dataset:
        raise NotImplementedError()

    def get_test_dataset(self, data_dir: str, data_file: str) -> Dataset:
        raise NotImplementedError()

    def get_labels(self) -> List[str]:
        return ["PLAUSIBLE", "NEUTRAL", "IMPLAUSIBLE"]

    def _create_examples(self, data_file_path: str, label_file_path: str, dataset_type: str) -> List[InputExample]:
        examples = []
        dataset = pd.read_csv(data_file_path, sep='\t', quoting=3)
        dataset = dataset.fillna("")
        if dataset_type is not 'test':
            label_set = pd.read_csv(label_file_path, sep='\t', header=None, names=['Id', 'Label'])
            for _, row in dataset.iterrows():
                for answer in range(1, 6):
                    examples.append(
                        InputExample(
                            id=f"{row['Id']}_{answer}",
                            title=row['Article title'],
                            prev_sentence=row['Previous context'],
                            now_sentence=row['Sentence'],
                            next_sentence=row['Follow-up context'],
                            answer=row[f'Filler{answer}'],
                            label=label_set.loc[f"{row['Id']}_{answer}"]
                        )
                    )
            return examples
        else:
            for _, row in dataset.iterrows():
                for answer in range(1, 6):
                    examples.append(
                        InputExample(
                            id=f"{row['Id']}_{answer}",
                            title=row['Article title'],
                            prev_sentence=row['Previous context'],
                            now_sentence=row['Sentence'],
                            next_sentence=row['Follow-up context'],
                            answer=row[f'Filler{answer}'],
                            label=None
                        )
                    )
            return examples

    def _convert_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        return convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=self.get_labels(),
            max_length=self.args.max_seq_length
        )

    def _create_dataset(self, data_file_path: str, label_file_path: str, dataset_type: str) -> TensorDataset:
        examples = self._create_examples(data_file_path, label_file_path, dataset_type)
        features = self._convert_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features], dtype=torch.long
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )
        return dataset


class SemEvalDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace, processor: DataProcessor) -> None:
        super().__init__()
        self.args = args
        self.processor = processor

    def prepare_data(self, dataset_type: str) -> Dataset:
        wandb_logger.log_text("Creating features from dataset file at {}".format(self.args.data_dir))

        if dataset_type == 'train':
            dataset = self.processor.get_train_dataset(self.args.data_dir, self.args.train_data, self.args.train_label)
        elif dataset_type == 'dev':
            dataset = self.processor.get_dev_dataset(self.args.data_dir, self.args.dev_data, self.args.dev_label)
        elif dataset_type == 'test':
            dataset = self.processor.get_test_dataset(self.args.data_dir, self.args.test_data)
        else:
            raise ValueError('Do not support')

        wandb_logger.log_text("Prepare {} dataset Count : {}".format(dataset_type, len(dataset)))
        return dataset

    def get_dataloader(self, dataset_type: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.prepare_data(dataset_type),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers
        )

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", self.args.train_batch_size, shuffle=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", self.args.dev_batch_size, shuffle=False)

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", self.args.test_batch_size, shuffle=False)

    def add_argparse_args(cls, parser: argparse.ArgumentParser, **kwargs) -> argparse.ArgumentParser:
        parser.add_argument('--data_dir', default='../data', type=str)

        parser.add_argument('--train_data', default='traindata.tsv', type=str, help='train data')
        parser.add_argument('--train_label', default='trainlables.tsv', type=str, help='train label')
        parser.add_argument('--dev_data', default='devdata.tsv', type=str, help='dev data')
        parser.add_argument('--dev_label', default='devlabels.tsv', type=str, help='dev label')
        parser.add_argument('--test_data', default='testdata.tsv', type=str, help='test data')

        parser.add_argument('--num_workers', default=16, type=int, help='passed to DataLoader')
        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--dev_batch_size', default=64, type=int)

        # args = parser.parse_args()
        # return args
        return parser


dm = SemEvalDataModule("distilbert-base-uncased")
dm.prepare_data()
dm.setup("fit")
next(iter(dm.train_dataloader()))