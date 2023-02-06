import os
import sys
import json
import torch
import evaluate
import jsonlines
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer
from mydatasets import NMTDataset
from models import Seq2SeqNMT
from torch.utils.data import DataLoader
from trainers import Trainer
from typing import List


@dataclass
class TrainingArguments:
    num_epochs: int = 1
    write_step: int = 1
    cuda_item: List = None
    tokenizer: object = None
    tgt_new2old: object = None

@dataclass
class ModelConfig:
    hidden_size: int = 100
    embed_size: int = 100
    src_vocab_size: int = None
    tgt_vocab_size: int = None
    encoder_layers: int = 1
    decoder_layers: int = 1
    src_padding_idx: int = 0
    tgt_padding_idx: int = 0


def load_jsonl_file(filepath):
    result = []
    with open(filepath) as f:
        for x in jsonlines.Reader(f):
            result.append(x)
    
    return result

class ComputeLoss:
    def __init__(self, ignore_index):
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def __call__(self, inputs, target):
        if inputs.dim() == 3:
            num_class = inputs.size(2)
            inputs = inputs.view(-1, num_class)
            target = target.view(-1)

        loss = self.loss_fn(inputs, target)

        return loss

class ComputeMetrics:
    def __init__(self):
        self.metrics = evaluate.load("sacrebleu")
    
    def __call__(self, predictions, references):
        metrics = self.metrics.compute(predictions=predictions, references=references)
        return metrics

def train_main():
    train_data_path = "../data/train_datasets.json"
    eval_data_path = "../data/eval_datasets.json"
    src_token_info_path = "../data/en_token_info.json"
    tgt_token_info_path = "../data/fr_token_info.json"
    pretrained_model_path = "../../pretrained_models/bert-base-multilingual-cased/"

    train_data = load_jsonl_file(train_data_path)
    eval_data = load_jsonl_file(eval_data_path)

    src_token_info = json.load(open(src_token_info_path))
    tgt_token_info = json.load(open(tgt_token_info_path))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    pad_token = tokenizer.pad_token
    src_pad_token_id = src_token_info["token2newId"][pad_token]
    tgt_pad_token_id = tgt_token_info["token2newId"][pad_token]

    hidden_size = 100
    embed_size = 100
    encoder_layers = 2
    decoder_layers = 2
    model_config = ModelConfig(
        hidden_size=hidden_size,
        embed_size=embed_size,
        src_vocab_size=len(src_token_info["token2oldId"]),
        tgt_vocab_size=len(tgt_token_info["token2oldId"]),
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        src_padding_idx=src_pad_token_id,
        tgt_padding_idx=tgt_pad_token_id,
    )

    model = Seq2SeqNMT(model_config)

    device = torch.device("cuda")
    model = model.to(device)

    max_length = 128
    ignore_index = -100

    train_dataset = NMTDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        src_old2new=src_token_info["old2new"],
        tgt_old2new=tgt_token_info["old2new"],
        ignore_index=ignore_index,
    )

    eval_dataset = NMTDataset(
        data=eval_data,
        tokenizer=tokenizer,
        max_length=max_length,
        src_old2new=src_token_info["old2new"],
        tgt_old2new=tgt_token_info["old2new"],
        ignore_index=ignore_index,
    )

    batch_size = 32

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.generate_batch,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_dataset.generate_batch,
    )

    num_epochs = 100
    write_step = 1
    cuda_item = ["encoder_input_ids", "decoder_input_ids", "labels"]
    training_args = TrainingArguments(
        num_epochs=num_epochs,
        write_step=write_step,
        cuda_item=cuda_item,
        tokenizer=tokenizer,
        tgt_new2old=tgt_token_info["new2old"],
    )

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    compute_loss = ComputeLoss(ignore_index=ignore_index)
    compute_metrics = ComputeMetrics()
    outputs_dir = "../train_outputs/"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        compute_loss=compute_loss,
        compute_metrics=compute_metrics,
        outputs_dir=outputs_dir,
        training_args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    os.chdir(sys.path[0])
    train_main()

