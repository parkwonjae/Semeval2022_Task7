from typing import Optional, List, Union
from inputdataclass import InputFeatures, InputExample
from transformers import PreTrainedTokenizer
from collections import defaultdict
import json

def convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None
):
    batch_encoding = tokenizer(
        [(example.prev_sentence+example.now_sentence+example.next_sentence) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs)
        features.append(feature)
    return features


def remove_symbol(sentence):
    sentence = sentence.replace("—", "")
    sentence = sentence.replace('\"', '')
    sentence = sentence.replace("(...)", "")
    sentence = sentence.replace("*", "")
    sentence = sentence.replace(";", "")
    return sentence


def standard_sentence(sentence):
    sentence = remove_symbol(sentence)
    sentence = sentence.strip().lower()
    return sentence


def construct_wikdict():
    wik_file = '../wiktionary/download-all-senses.json'
    wik_dict = defaultdict(list)

    with open(wik_file, 'r', encoding='utf-8') as f:
        counter = 0
        for line in f:
            if counter % 100000 == 0:
                print(counter)
            data = json.loads(line)
            data.pop('pronunciations', 'no')
            data.pop('lang', 'no')
            data.pop('translations', 'no')
            data.pop('sounds', 'no')
            wik_dict[data['word']].append(data)
            counter += 1

    # join lower case with upper case
    new_wik_dict = {}
    for word in wik_dict:
        if word.lower() not in new_wik_dict:
            new_wik_dict[word.lower()] = wik_dict[word]
        elif word.lower() == word:
            new_wik_dict[word.lower()] = wik_dict[word] + new_wik_dict[word.lower()]
        else:
            new_wik_dict[word.lower()] = new_wik_dict[word.lower()] + wik_dict[word]

    wik_dict = new_wik_dict
    print(wik_dict)

construct_wikdict()