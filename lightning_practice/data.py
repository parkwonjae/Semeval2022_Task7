from datetime import datetime
from typing import Optional, List
import datasets
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import pandas as pd
from inputdataclass import InputFeatures, InputExample
from utils import convert_examples_to_features, standard_sentence


class GLUEDataModule(LightningDataModule):

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


# 사용방법
# dm = GLUEDataModule("distilbert-base-uncased")
# dm.prepare_data()
# dm.setup("fit")
# next(iter(dm.train_dataloader()))


class SemEvalDataModule(LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str,
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_labels = 3
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def _create_examples(self, dataset: pd.DataFrame):
        examples = []
        dataset = dataset.fillna("")
        for _, row in dataset.iterrows():
            for answer in range(1, 6):
                examples.append(
                    InputExample(
                        id=f"{row['Id']}_{answer}",
                        title=standard_sentence(row['Article title']),
                        prev_sentence=standard_sentence(row['Previous context']),
                        now_sentence=standard_sentence(
                            row['Sentence'].replace("______", row[f'Filler{answer}'])
                        ),
                        next_sentence=standard_sentence(row['Follow-up context']),
                        answer=row[f'Filler{answer}'],
                ))
        return examples

    def _convert_features(self, examples: List[InputExample]):
        return convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.max_seq_length
        )

    def _create_dataset(self, dataset: pd.DataFrame, stage: Optional[str]):
        examples = self._create_examples(dataset)
        features = self._convert_features(examples)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features], dtype=torch.long
        )

        label_str_to_int_map = {'PLAUSIBLE': 0, 'NEUTRAL': 1, 'IMPLAUSIBLE': 2}

        if 'train' in stage:
            labelset = pd.read_csv('../data/train/trainlabels.tsv', sep='\t', header=None, names=['Id', 'Label'])
            labelset = list(labelset["Label"])
            labelset = [label_str_to_int_map[label] for label in labelset]
            all_label = torch.tensor(labelset, dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids, all_label
            )
            print()
            print('**** train dataset ****')
            print('all_input_ids: {}\n'.format(all_input_ids[:2]))
            print('all_attention_mask: {}\n'.format(all_attention_mask[:2]))
            print('all_token_type_ids: {}\n'.format(all_token_type_ids[:2]))
            print('all_label: {}\n'.format(all_label[:2]))
            return dataset

        elif 'val' in stage:
            labelset = pd.read_csv('../data/dev/devlabels.tsv', sep='\t', header=None, names=['Id', 'Label'])
            labelset = list(labelset["Label"])
            labelset = [label_str_to_int_map[label] for label in labelset]
            all_label = torch.tensor(labelset, dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids, all_label
            )
            print()
            print('**** val dataset ****')
            print('all_input_ids: {}\n'.format(all_input_ids[:2]))
            print('all_attention_mask: {}\n'.format(all_attention_mask[:2]))
            print('all_token_type_ids: {}\n'.format(all_token_type_ids[:2]))
            print('all_label: {}\n'.format(all_label[:2]))
            return dataset

        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids
        )
        return dataset

    #멀티 프로세스로 데이터를 다운받으면 데이터가 꼬임. 그래서 하나의 프로세스로만 되도록 픽스.
    # download, tokenize 작업 등
    # self.x =y 같은 할당을 여기서 하는건 추천하지 않음. main process 에서 부름
    # def prepare_data(self, dataset_type: str) -> List[InputExample]:
    #     if dataset_type == 'train':
    #         dataset = pd.read_csv('../data/train/traindata.tsv', sep='\t', quoting=3)
    #         self.train_data = self._create_examples(dataset)
    #     elif dataset_type == 'val':
    #         dataset = pd.read_csv('../data/dev/devdata.tsv')
    #     elif dataset_type == 'test':
    #         dataset = pd.read_csv('../data/test/testdata.tsv')

    # 모든 GPU에서 수행하려는 데이터 작업.
    # 클래스 수 카운트, 보캡 빌드, train/val/test 데이터 분리해서 형성, 데이터셋 만들기, apply transform
    # 이 메서드에서는 stage 인수가 필요. trainer setup 로직을 분리하는데 사용. trainer.{fit, validate, test, predict}
    # stage=None 일때, 모든 단계가 설정되었다고 가정,
    def setup(self, stage: Optional[str] = None):
        # # Assign Train/val split(s) for use in Dataloaders
        # if stage in (None, "fit"):
        #     mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        #     self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        #
        # # Assign Test split(s) for use in Dataloaders
        # if stage in (None, "test"):
        #     self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)
        if stage in (None, 'fit'):
            train_dataset = pd.read_csv('../data/train/traindata.tsv', sep='\t', quoting=3)
            self.train_data = self._create_dataset(train_dataset, 'train')
            val_dataset = pd.read_csv('../data/dev/devdata.tsv', sep='\t', quoting=3)
            self.val_data = self._create_dataset(val_dataset, 'val')

        if stage in (None, 'test'):
            test_dataset = pd.read_csv('../data/test/testdata.tsv', sep='\t', quoting=3)
            self.test_data = self._create_dataset(test_dataset)

    # train_dataloader 생성 메소드. setup에서 정의한 데이터셋을 래핑. Trainer.fit()메서드가 사용하는 데이터로더.
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.train_batch_size, sampler=RandomSampler(self.train_data))

    # val_dataloader해서 유효성 검사 데이터 로더를 생성. setup에서 정의한 데이터 세트를 래핑. fit()과 validate()메소드가 사용하는 데이터 로더.
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.eval_batch_size, sampler=SequentialSampler(self.val_data))

    # 테스트 데이터 로더 생성. setup에서 정의한ㄷ ㅔ이터 세트를 래핑 test()메서드를 사용
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.test_batch_size,  num_workers=16)





# dm = SemEvalDataModule("distilbert-base-uncased")
# dm.setup('fit')
# for batch in dm.train_dataloader():
#     print(batch)



'''
Using a DataModule


dm = MNISTDataModule()
model = Model()
trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)
trainer.validate(datamodule=dm)
trainer.predict(datamodule=dm)


dm = MNISTDataModule()
dm.prepare_data()
dm.setup(stage="fit")
model = Model(num_classes=dm.num_classes, width=dm.width, vocab=dm.vocab)
trainer.fit(model, dm)
dm.setup(stage="test")
trainer.test(datamodule=dm)

'''


'''
라이트닝 없이 데이터모듈 사용
# download, etc...
dm = MNISTDataModule()
dm.prepare_data()

# splits/transforms
dm.setup(stage="fit")

# use data
for batch in dm.train_dataloader():
    ...

for batch in dm.val_dataloader():
    ...

dm.teardown(stage="fit")

# lazy load test data
dm.setup(stage="test")
for batch in dm.test_dataloader():
    ...

dm.teardown(stage="test")
'''

