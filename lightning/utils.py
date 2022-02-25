from typing import Optional, Dict, List, Union
from lightning.inputdataclass import InputFeatures, InputExample
from transformers import PreTrainedTokenizer
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(name='test', project='pytorchlightning')

def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    label_list: Optional[List[str]],
    max_length: Optional[int] = None
) -> List[InputFeatures]:

    if max_length is None:
        max_length = tokenizer.max_length
    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None, List[int]]:
        if example.label is None:
            return None
        else:
            return label_map[example.label]

    labels = [label_from_example(example) for example in examples]
    batch_encoding = tokenizer(
        [(example.title) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)
    for i, example in enumerate(examples[:5]):
        wandb_logger.log_text("*** EXAMPLE ***")
        wandb_logger.log_text("id : {}".format(example.id))
        wandb_logger.log_text("features: {}".format(features[i]))
    return features