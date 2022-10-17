from dataclasses import dataclass
import json
import os
import re
import sys
import argparse
import logging
from collections import namedtuple
from typing import Dict

import torch
import torch.distributed as dist
from torch import nn
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import SGD
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter

sys.path.append(os.environ["HOME"])
from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.truncate_dataset import TruncateDataset
from datasets.collate_functions import collate_to_max_length
from models.bert_query_ner import BertQueryNER
from models.model_config import BertQueryNerConfig
from utils.misc import get_parser, set_random_seed

set_random_seed(0)

def init_ddp(dist, device='cuda', init_method='env://'):
    assert device in ["cpu", "cuda"]

    if device=="cpu":
        backend = "gloo"
    elif device=="cuda":
        backend = "nccl"

    dist.init_process_group(
    backend=backend,  # 可设置{'nccl','gloo'},GPU使用nccl,CPU使用gloo
    init_method=init_method
    )

    global_rank=dist.get_rank()   # 从进程组设置中自动获取rank值
    world_size=dist.get_world_size()
    return global_rank,world_size

def set_data_distributed_training():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['NCCL_SOCKET_IFNAME'] = 'eno1'
    
    # 获取global_rank 和 world_size
    global_rank,world_size = init_ddp(dist)
    device = int(os.environ['LOCAL_RANK']) #ddp更换device
    torch.cuda.set_device(device)

    return global_rank, world_size, device

def close_data_distributed_training():
    dist.destroy_process_group()

def query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels, flat=False):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        start_label_mask: [bsz, seq_len]
        end_label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
        flat: if True, decode as flat-ner
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()
    output = dict()
    output["tp"] = tp
    output["fp"] = fp
    output["fn"] = fn
    return output

class QuerySpanF1(nn.Module):
    def __init__(self, flat=False):
        self.flat = flat

    def forward(self, start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels):
        return query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels,
                             flat=self.flat)

class MRCNERTrainer:
    def __init__(self, config):
        self.bce_loss = BCEWithLogitsLoss(reduction="none")
        self.span_loss_candidates = "pred_and_gold"
        self.num_epochs = 20
        self.weight_start = 1.0
        self.weight_end = 1.0
        self.weight_span = 0.1
        self.flat_ner = False
        self.span_f1 = query_span_f1
        self.specific_device = config.specific_device
        self.training_mode = config.training_mode

    def train(self, model, optimizer, scheduler, train_dataloader, eval_dataloader, writer, model_dir):

        if self.specific_device is not None:
            device = self.specific_device
        else:
            device = self.cal_gpu(model)
        
        outer_progressbar = tqdm(range(self.num_epochs), leave=False)
        for epoch in range(self.num_epochs):
            model.train()
            tp = 0
            fp = 0
            fn = 0
            train_loss = 0
            train_start_loss = 0
            train_end_loss = 0
            train_match_loss = 0

            inner_progressbar = tqdm(range(len(train_dataloader)), leave=False)
            for batch in train_dataloader:
                batch = [x.to(device) for  x in batch]
                tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch

                # num_tasks * [bsz, length, num_labels]
                attention_mask = (tokens != 0).long()
                start_logits, end_logits, span_logits = model(tokens, attention_mask, token_type_ids)

                start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                                    end_logits=end_logits,
                                                                    span_logits=span_logits,
                                                                    start_labels=start_labels,
                                                                    end_labels=end_labels,
                                                                    match_labels=match_labels,
                                                                    start_label_mask=start_label_mask,
                                                                    end_label_mask=end_label_mask
                                                                    )

                loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                start_preds, end_preds = start_logits > 0, end_logits > 0
                span_f1_stats = self.span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                            start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                            match_labels=match_labels)
                
                tp += span_f1_stats["tp"]
                fp += span_f1_stats["fp"]
                fn += span_f1_stats["fn"]

                train_loss += loss.item()
                train_start_loss += start_loss.item()
                train_end_loss += end_loss.item()
                train_match_loss += match_loss.item()

                inner_progressbar.update(1)

            train_loss /= len(train_dataloader)
            train_start_loss /= len(train_dataloader)
            train_end_loss /= len(train_dataloader)
            train_match_loss /= len(train_dataloader)

            span_recall = tp / (tp + fn + 1e-10)
            span_precision = tp / (tp + fp + 1e-10)
            span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)

            writer.add_scalar("train/loss/loss", train_loss, global_step=epoch)
            writer.add_scalar("train/loss/start_loss", train_start_loss, global_step=epoch)
            writer.add_scalar("train/loss/end_loss", train_end_loss, global_step=epoch)
            writer.add_scalar("train/loss/match_loss", train_match_loss, global_step=epoch)

            writer.add_scalar("train/metric/recall", span_recall, global_step=epoch)
            writer.add_scalar("train/metric/precision", span_precision, global_step=epoch)
            writer.add_scalar("train/metric/f1", span_f1, global_step=epoch)

            eval_result = self.evaluate(model, eval_dataloader)

            writer.add_scalar("eval/loss/loss", eval_result["loss"], global_step=epoch)
            writer.add_scalar("eval/loss/start_loss", eval_result["start_loss"], global_step=epoch)
            writer.add_scalar("eval/loss/end_loss", eval_result["end_loss"], global_step=epoch)
            writer.add_scalar("eval/loss/match_loss", eval_result["match_loss"], global_step=epoch)

            writer.add_scalar("eval/metric/recall", eval_result["span_recall"], global_step=epoch)
            writer.add_scalar("eval/metric/precision", eval_result["span_precision"], global_step=epoch)
            writer.add_scalar("eval/metric/f1", eval_result["span_f1"], global_step=epoch)

            outer_progressbar.update(1)

            p = eval_result["span_precision"]
            r = eval_result["span_recall"]
            f1 = eval_result["span_f1"]
            model_name = f"model_for_conll03_epoch_{epoch}_p_{p:.4f}_r_{r:.4f}_f1_{f1:.4f}.pth"
            model_path = os.path.join(model_dir, model_name)
            if self.training_mode != "ddp" or dist.get_rank() == 0:
                torch.save(model.state_dict(), model_path)
            

    def evaluate(self, model, eval_dataloader):
        if self.specific_device is not None:
            device = self.specific_device
        else:
            device = self.cal_gpu(model)

        model.eval()
        eval_loss = 0
        eval_start_loss = 0
        eval_end_loss = 0
        eval_match_loss = 0
        tp = 0
        fp = 0
        fn = 0
        with torch.no_grad():
            progressbar = tqdm(range(len(eval_dataloader)), leave=False)
            for batch in eval_dataloader:
                batch = [x.to(device) for x in batch]
                tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch

                attention_mask = (tokens != 0).long()
                start_logits, end_logits, span_logits = model(tokens, attention_mask, token_type_ids)

                start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                                    end_logits=end_logits,
                                                                    span_logits=span_logits,
                                                                    start_labels=start_labels,
                                                                    end_labels=end_labels,
                                                                    match_labels=match_labels,
                                                                    start_label_mask=start_label_mask,
                                                                    end_label_mask=end_label_mask
                                                                    )

                total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

                eval_loss += total_loss.item()
                eval_start_loss += start_loss.item()
                eval_end_loss += end_loss.item()
                eval_match_loss += match_loss.item()

                start_preds, end_preds = start_logits > 0, end_logits > 0
                span_f1_stats = self.span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                            start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                            match_labels=match_labels)
                
                tp += span_f1_stats["tp"]
                fp += span_f1_stats["fp"]
                fn += span_f1_stats["fn"]

                progressbar.update(1)

        eval_loss /= len(eval_dataloader)
        eval_start_loss /= len(eval_dataloader)
        eval_end_loss /= len(eval_dataloader)
        eval_match_loss /= len(eval_dataloader)

        span_recall = tp / (tp + fn + 1e-10)
        span_precision = tp / (tp + fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)

        output = dict()
        output["loss"] = total_loss
        output["start_loss"] = start_loss
        output["end_loss"] = end_loss
        output["match_loss"] = match_loss
        output["span_recall"] = span_recall
        output["span_precision"] = span_precision
        output["span_f1"] = span_f1

        return output
                

    def compute_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
        batch_size, seq_len = start_logits.size()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            elif self.span_loss_candidates == "pred_gold_random":
                gold_and_pred = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
                data_generator = torch.Generator()
                data_generator.manual_seed(0)
                random_matrix = torch.empty(batch_size, seq_len, seq_len).uniform_(0, 1)
                random_matrix = torch.bernoulli(random_matrix, generator=data_generator).long()
                random_matrix = random_matrix.cuda()
                match_candidates = torch.logical_or(
                    gold_and_pred, random_matrix
                )
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()

        start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

        return start_loss, end_loss, match_loss

    def cal_gpu(self, module):
        """
        判断模型参数所处的显卡，只适用于模型所有结构在相同卡
        
        """
        for name, param in module.named_parameters():
            return param.device


def configure_optimizers(model, train_dataloader, optimizer="torch.adam", weight_decay=0.01, 
                         lr=3e-5, lr_mini=3e-7, gpus="0", accumulate_grad_batches=4, 
                         max_epochs=20, lr_scheduler="polydecay", warmup_steps=0, 
                         final_div_factor=1e4):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if optimizer == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=lr)
    elif optimizer == "torch.adam":
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=lr,
                                      weight_decay=weight_decay)
    else:
        optimizer = SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    num_gpus = len([x for x in str(gpus).split(",") if x.strip()])
    t_total = (len(train_dataloader) // (accumulate_grad_batches * num_gpus) + 1) * max_epochs
    if lr_scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, pct_start=float(warmup_steps/t_total),
            final_div_factor=final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
    elif lr_scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif lr_scheduler == "polydecay":
        if lr_mini == -1:
            lr_mini = lr / 5
        else:
            lr_mini = lr_mini
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total, lr_end=lr_mini)
    else:
        raise ValueError
    return optimizer, scheduler


@dataclass
class ModelConfig:
    hidden_size: int = 768
    mrc_dropout: float = 0.1
    classifier_intermediate_hidden_size: int = 2048
    pretrained_model_path: str = None

@dataclass
class TrainerConfig:
    specific_device: torch.device = None
    training_mode: str = None

def get_dataloader(prefix="train", batch_size=10, data_dir=None, 
                   bert_dir=None, limit: int = None,
                   max_length=200, is_chinese=False,
                   num_workers=8) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        json_path = os.path.join(data_dir, f"mrc-ner.{prefix}")
        vocab_path = os.path.join(bert_dir, "vocab.txt")
        dataset = MRCNERDataset(json_path=json_path,
                                tokenizer=BertWordPieceTokenizer(vocab_path),
                                max_length=max_length,
                                is_chinese=is_chinese,
                                pad_to_maxlen=False
                                )

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length
        )

        return dataloader

def train_main():
    os.chdir(sys.path[0])

    bert_config_dir = "../../../pretrained_model/bert-large-cased/"
    data_dir = "../../data/MRC/conll03/"
    train_data_path = "../../data/MRC/conll03/mrc-ner.train"
    eval_data_path = "../../data/MRC/conll03/mrc-ner.dev"
    vocab_path = os.path.join(bert_config_dir, "vocab.txt")
    training_mode = "single" # single | multi | ddp

    batch_size = 8
    max_length = 200
    is_chinese = False
    num_workers = 8
    limit = None
    tokenizer = BertWordPieceTokenizer(vocab_path)

    train_dataset = MRCNERDataset(
        json_path=train_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_chinese=is_chinese,
        pad_to_maxlen=False
    )

    eval_dataset = MRCNERDataset(
        json_path=eval_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_chinese=is_chinese,
        pad_to_maxlen=False
    )

    if limit is not None:
        train_dataset = TruncateDataset(train_dataset, limit)
        eval_dataset = TruncateDataset(eval_dataset, limit)

    global_rank, world_size, device = None, None, torch.device("cuda")
    train_sampler, eval_sampler = None, None
    if training_mode == "ddp":
        global_rank, world_size, device = set_data_distributed_training()
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    
    shuffle = train_sampler is None
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_to_max_length,
        sampler=train_sampler
    )    

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_to_max_length,
        sampler=eval_sampler
    )

    tensorboard_path = "../outputs/tensorboard/"
    model_dir = "../outputs/train_model/"
    writer = SummaryWriter(tensorboard_path)
    
    bert_dropout = 0.1
    mrc_dropout = 0.3
    classifier_act_func = "gelu"
    classifier_intermediate_hidden_size = 2048
    bert_config = BertQueryNerConfig.from_pretrained(bert_config_dir,
                                                     hidden_dropout_prob=bert_dropout,
                                                     attention_probs_dropout_prob=bert_dropout,
                                                     mrc_dropout=mrc_dropout,
                                                     classifier_act_func=classifier_act_func,
                                                     classifier_intermediate_hidden_size=classifier_intermediate_hidden_size)

    model = BertQueryNER.from_pretrained(bert_config_dir, config=bert_config)

    gpus = "0,1"
    device = torch.device("cuda")
    lr = 3e-5
    lr_mini = 3e-7
    optimizer, scheduler = configure_optimizers(model=model, train_dataloader=train_dataloader, gpus=gpus,
                                                lr=lr, lr_mini=lr_mini)

    specific_device = None
    if training_mode != "ddp":
        specific_device = device
    
    trainer_config = TrainerConfig(
        specific_device=specific_device,
        training_mode=training_mode
    )
    trainer = MRCNERTrainer(trainer_config)

    if training_mode == "single":
        model = model.to(device)
    elif training_mode == "multi":
        model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)
    elif training_mode == "ddp":
        model = DDP(model.cuda(), device_ids=[device], find_unused_parameters=True)

    trainer.train(model, optimizer, scheduler, train_dataloader, eval_dataloader, writer, model_dir)

    if training_mode == "ddp":
        close_data_distributed_training()

    """torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="192.168.1.224" --master_port=9997 mrc_ner_train.py"""

if __name__ == "__main__":
    os.chdir(sys.path[0])
    train_main()





