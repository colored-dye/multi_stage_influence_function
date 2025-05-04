import os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer
from datasets import load_dataset

from modeling_gpt_neox import GPTNeoXForCausalLM

from loguru import logger


"""dolly instruction/response format."""

INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)
PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

def dolly_format_with_response(rec):
    instruction = rec['instruction']
    response = rec['response']
    context = rec.get('context')
    if context:
        text = PROMPT_WITH_INPUT_FORMAT.format(
            instruction=instruction, response=response, input=context)
    else:
        text = PROMPT_NO_INPUT_FORMAT.format(
            instruction=instruction, response=response)
    return text


def dolly_get_labels(input_ids: torch.Tensor, tokenizer: AutoTokenizer):
    labels = input_ids.clone()
    labels[labels==tokenizer.pad_token_id] == -100
    response_token_ids = tokenizer.encode(RESPONSE_KEY, return_tensors='pt')
    for i in range(labels.shape[0]):
        response_token_ids_start_idx = None
        for idx in torch.where(labels[i] == response_token_ids[0])[0]:
            response_token_ids_start_idx = idx
            break
        response_token_ids_end_idx = response_token_ids_start_idx + 1
        labels[i, :response_token_ids_end_idx] = -100
    return labels


def load_partial_pile(pile_path, file_no, percent_begin, percent_end):
    """
    Attributes:
        percent_begin: 0-100.
        percent_end: 0-100.
    """
    return load_dataset(pile_path,
                        data_files=f"train/{file_no:02}.jsonl.zst",
                        split=f"train[{percent_begin}%:{percent_end}%]",
                        cache_dir="./cache")
