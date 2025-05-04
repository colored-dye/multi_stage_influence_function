from tqdm import tqdm
import os
import time
from typing import Iterable
import pickle
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modeling_gpt_neox import GPTNeoXForCausalLM
from transformers import (
    AutoTokenizer, AutoConfig, default_data_collator
)
from datasets import load_dataset

from trak import TRAKer
from trak.modelout_functions import AbstractModelOutput

import utils


# device = "cuda:2"
device = "cuda"
pretrained_model_path = "/storage/Model/EleutherAI/pythia-70m"
max_length = 512
ckpt_dir = "trak_models/"
n_ckpts = 10
batch_size = 48
save_dir = "infls/trak/"
trak_proj_dim = 4096


class CustomModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


# class CustomModelOutput(AbstractModelOutput):
#     @staticmethod
#     def get_output(model, weights, buffers, input_ids, attention_mask, label):
#         # kw_inputs = {
#         #     "input_ids": input_ids,
#         #     "attention_mask": attention_mask,
#         # }
#         input_ids = input_ids.unsqueeze(0)
#         attention_mask = attention_mask.unsqueeze(0)
#         outputs = torch.func.functional_call(model, (weights, buffers), args=(input_ids, attention_mask))
#         logits = outputs.logits
#         logits = logits.reshape(-1, logits.size(-1))
#         label = label.reshape(-1)
#         loss = F.cross_entropy(logits, label, reduction="sum")
#         return loss

#     @staticmethod
#     def get_out_to_loss_grad(model, weights, buffers, batch: Iterable[torch.Tensor]) -> torch.Tensor:
#         input_ids, attention_mask, labels = batch
#         # kw_inputs = {
#         #     "input_ids": input_ids,
#         #     "attention_mask": attention_mask,
#         # }
#         outputs = torch.func.functional_call(model, (weights, buffers), args=(input_ids, attention_mask))
#         logits = outputs.logits
#         logits_dim = logits.size(-1)
#         batch_size = logits.size(0)
#         logits = logits.reshape(-1, logits.size(-1))
#         labels = labels.reshape(-1)
#         out_grads = torch.func.grad(lambda logits, labels: F.cross_entropy(logits, labels, reduction='sum'))(logits, labels)
#         out_grads = out_grads.reshape(batch_size, -1, logits_dim)
#         # out_grads = torch.sum(out_grads, dim=1, keepdim=True)
#         print(out_grads.shape)
#         return out_grads.clone().detach()


class CustomModelOutput(AbstractModelOutput):
    """Margin for text classification models. This assumes that the model takes
    in input_ids, token_type_ids, and attention_mask.

    .. math::

        \\text{logit}[\\text{correct}] - \\log\\left(\\sum_{i \\neq
        \\text{correct}} \\exp(\\text{logit}[i])\\right)

    """

    def __init__(self, temperature=1.0) -> None:
        super().__init__()
        self.softmax = torch.nn.Softmax(-1)
        self.loss_temperature = temperature

    @staticmethod
    def get_output(
        model,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        input_id: Tensor,
        attention_mask: Tensor,
        label: Tensor,
    ) -> Tensor:
        kw_inputs = {
            "input_ids": input_id.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }

        # print(label)

        logits = torch.func.functional_call(
            model, (weights, buffers), args=(), kwargs=kw_inputs
        )
        bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, torch.arange(len(label)), label]
        # logits_correct = logits
        # print(logits.shape, logits_correct.shape)

        cloned_logits = logits.clone()
        # print(cloned_logits.shape, label.shape, label)
        cloned_logits[bindex, torch.arange(len(label)), label] = -torch.inf
        # print(cloned_logits.shape)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()

    @staticmethod
    def get_out_to_loss_grad(
        model, weights, buffers, batch: Iterable[Tensor]
    ) -> Tensor:
        input_ids, attention_mask, labels = batch
        kw_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        logits = torch.func.functional_call(
            model, (weights, buffers), args=(), kwargs=kw_inputs
        )
        batch_indices = torch.arange(logits.shape[0]).unsqueeze(1)  # Shape: (48, 1)
        sequence_indices = torch.arange(logits.shape[1]).unsqueeze(0)  # Shape: (1, 511)

        ps = torch.softmax(logits, dim=-1)[batch_indices, sequence_indices, labels]
        # ps = torch.softmax(logits, dim=-1)[
        #     torch.arange(labels.shape[0]), labels
        # ]
        return (1 - ps).clone().detach().sum(-1).unsqueeze(-1)


def tokenize_and_pad(examples):
    return tokenizer(examples['sentence'], max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')


def collate_fn(batch):
    return (
        torch.stack(([torch.tensor(x['input_ids'][:-1]) for x in batch])),
        torch.stack(([torch.tensor(x['attention_mask'][:-1]) for x in batch])),
        torch.stack(([torch.tensor(x['input_ids'][1:]) for x in batch])),
    )


def featurize(traker: TRAKer, model, train_loader):
    time_featurize = 0
    for model_id in tqdm(range(n_ckpts), desc="Checkpoints"):
        _ckpt = torch.load(os.path.join(ckpt_dir, f"{model_id}.ckpt"), map_location=device)
        model.model.load_state_dict(_ckpt)
        traker.load_checkpoint(model.state_dict(), model_id=model_id)

        start = time.time()
        for batch in tqdm(train_loader, desc=f"Ckpt [{model_id}]"):
            batch = [x.to(device) for x in batch]
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        end = time.time()
        time_featurize += end-start

    start = time.time()
    traker.finalize_features()
    end = time.time()
    time_featurize += end-start
    print(f"Time on featurizing: {time_featurize:.3f} s")


def scoring(traker, model, test_loader):
    time_score = 0
    for model_id in range(n_ckpts):
        ckpt = torch.load(os.path.join(ckpt_dir, f"{model_id}.ckpt"), map_location=device)
        model.model.load_state_dict(ckpt)

        start = time.time()
        traker.start_scoring_checkpoint(exp_name="test",
                                        checkpoint=model.state_dict(),
                                        model_id=model_id,
                                        num_targets=len(ds['test']))

        for batch in tqdm(test_loader):
            batch = [x.to(device) for x in batch]
            traker.score(batch, num_samples=batch[0].shape[0])
        end = time.time()
        time_score += end-start

    start = time.time()
    scores = traker.finalize_scores(exp_name="test")
    end = time.time()
    time_score += end-start
    print(f"Time on scoring: {end-start:.3f} s")
    return scores


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = load_dataset("/storage/Dataset/ptb_text_only", trust_remote_code=True)
    with open("choices_candidates.pkl", "rb") as fp:
        choices_candidates = pickle.load(fp)
    with open("choices_queries.pkl", "rb") as fp:
        choices_queries = pickle.load(fp)
    choices_candidates = np.array(choices_candidates)
    ds_candidates = ds['train'].select(choices_candidates.ravel().tolist())
    ds_queries = ds['test'].select(choices_queries)
    ds['train'] = ds_candidates
    ds['test'] = ds_queries
    
    tokenized_ds = ds.map(tokenize_and_pad, batched=True, remove_columns=['sentence'])

    config = AutoConfig.from_pretrained("config.json")
    config.use_hook = False
    _model = GPTNeoXForCausalLM(config=config)
    _model = _model.to(device)
    model = CustomModel(_model)

    traker = TRAKer(model=model,
                    save_dir=f"trak_results-{trak_proj_dim}",
                    task=CustomModelOutput,
                    train_set_size=len(ds['train']),
                    device=device,
                    proj_dim=trak_proj_dim,
                    use_half_precision=False,
                    proj_max_batch_size=32)

    train_loader = DataLoader(tokenized_ds['train'], batch_size, shuffle=False, collate_fn=collate_fn)

    featurize(traker, model, train_loader)

    test_loader = DataLoader(tokenized_ds['test'], batch_size, shuffle=False, collate_fn=collate_fn)

    scores = scoring(traker, model, test_loader)
    with open("trak_scores.pkl", "wb") as fp:
        pickle.dump(scores, fp)

    os.makedirs(save_dir, exist_ok=True)

    n_queries = len(choices_queries)
    n_candidates_per_query = len(choices_candidates[0])

    for i in range(n_queries):
        infls = scores[i*n_candidates_per_query:(i+1)*n_candidates_per_query, i].tolist()
        with open(os.path.join(save_dir, f"{i}.pkl"), "wb") as fp:
            pickle.dump(infls, fp)
