
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Callable
from transformers import AutoTokenizer, AutoModel
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass

from models.model import GlobalPointer
from datasets.dataset import CLUENERDataset, collate_fn_for_clue_ner

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()

def compute_loss(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


def compute_metrics_for_batch(y_pred: np.ndarray, y_true: np.ndarray):
    pred = []
    true = []
    for b, l, start, end in zip(*np.where(y_pred>0)):
        pred.append((b, l, start, end))
    for b, l, start, end in zip(*np.where(y_true>0)):
        true.append((b, l, start, end))

    R = set(pred)
    T = set(true)
    X = len(R & T)
    Y = len(R)
    Z = len(T)
    
    # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    pred_set = set(pred)
    true_set = set(true)
    TP = len(pred_set.intersection(true_set))
    FP = len(pred_set.difference(true_set))
    FN = len(true_set.difference(pred_set))
    
    outputs = dict()
    outputs["TP"] = TP
    outputs["FP"] = FP
    outputs["FN"] = FN
    if TP + FP != 0:
        outputs["precision"] = TP / (TP + FP)
    else:
        outputs["precision"] = 0
    
    if TP + FN != 0:
        outputs["recall"] = TP / (TP + FN)
    else:
        outputs["recall"] = 0

    if outputs["precision"] + outputs["recall"] != 0:
        outputs["f1"] = 2 * outputs["precision"] * outputs["recall"] / (outputs["precision"] + outputs["recall"])

    return outputs

def compute_metrics(preds_list: List[np.ndarray], labels_list: List[np.ndarray]):
    TP = 0
    FP = 0
    FN = 0
    for preds, labels in zip(preds_list, labels_list):
        outputs = compute_metrics_for_batch(preds, labels)
        TP += outputs["TP"]
        FP += outputs["FP"]
        FN += outputs["FN"]
    
    precision = 0
    if TP + FP != 0:
        precision = TP / (TP + FP)
    recall = 0
    if TP + FN != 0:
        recall = TP / (TP + FN)
    f1 = 0
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)

    outputs = dict()
    outputs["TP"] = TP
    outputs["FP"] = FP
    outputs["FN"] = FN
    outputs["precision"] = precision
    outputs["recall"] = recall
    outputs["f1"] = f1

    return outputs


class Trainer:
    def __init__(self, config=None):
        if config is not None:
            self.num_epochs = config.num_epochs
            self.cuda_item = config.cuda_item
            self.compute_loss = config.compute_loss
            self.compute_metrics = config.compute_metrics
            self.tokenizer = config.tokenizer
            self.entity_id2type = config.entity_id2type

    def train(self, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader):
        device = None
        for _, p in model.named_parameters():
            device = p.device
            break

        for epoch in tqdm(range(self.num_epochs)):
            model.train()

            train_loss = 0
            preds_list = []
            labels_list = []

            for batch in tqdm(train_dataloader, total=len(train_dataloader)):
                inputs = {k: v.to(device) for k, v in batch.items() if k in self.cuda_item}
                logits = model(inputs)

                loss = self.compute_loss(logits, inputs["labels"])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                train_loss += loss.item()
                preds_list.append(logits.detach().cpu().numpy())
                labels_list.append(batch["labels"].detach().cpu().numpy())

            train_loss /= len(train_dataloader)
            train_metrics = self.compute_metrics(preds_list, labels_list)

            eval_loss, eval_metrics = self.evaluation(model, eval_dataloader)

            print(f"Epoch {epoch}, train loss: {train_loss:.6f}, eval loss: {eval_loss:.6f}, train f1: {train_metrics['f1']}, eval f1: {eval_metrics['f1']}")

            # decode_result = inference(model, eval_dataloader, self.entity_id2type)

    def evaluation(self, model, eval_dataloader):
        model.eval()
        device = None
        for _, p in model.named_parameters():
            device = p.device
            break
        eval_loss = 0
        preds_list = []
        labels_list = []
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                inputs = {k: v.to(device) for k, v in batch.items() if k in self.cuda_item}
                logits = model(inputs)

                loss = self.compute_loss(logits, inputs["labels"])
                
                eval_loss += loss.item()
                preds_list.append(logits.detach().cpu().numpy())
                labels_list.append(batch["labels"].detach().cpu().numpy())

        eval_loss /= len(eval_dataloader)
        eval_metrics = self.compute_metrics(preds_list, labels_list)

        return eval_loss, eval_metrics


def decode_single(text, pred_matrix, offset_mapping, entity_id2type, threshold=0):
    ent_list = {}
    for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = entity_id2type[ent_type_id]
        ent_char_span = [offset_mapping[token_start_index][0], offset_mapping[token_end_index][1]]
        ent_text = text[ent_char_span[0]:ent_char_span[1]]

        ent_type_dict = ent_list.get(ent_type, {})
        ent_text_list = ent_type_dict.get(ent_text, [])
        ent_text_list.append(ent_char_span)
        ent_type_dict.update({ent_text: ent_text_list})
        ent_list.update({ent_type: ent_type_dict})
    return ent_list

def decode_batch(logits, text_list, offset_mapping_list, entity_id2type):
    batch_size = logits.shape[0]
    text_entities = []
    for idx in range(batch_size):
        text = text_list[idx]
        pred_matrix = logits[idx]
        offset_mapping = offset_mapping_list[idx]
        entities = decode_single(text, pred_matrix, offset_mapping, entity_id2type)
        text_entities.append({"text": text, "entities": entities})
    
    return text_entities


def inference(model, dataloader, entity_id2type):
    model.eval()
    cuda_item = ["input_ids", "attention_mask", "token_type_ids"]
    device = None
    for _, p in model.named_parameters():
        device = p.device
        break

    decode_result = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs = {k: v.to(device) for k, v in batch.items() if k in cuda_item}
            logits = model(inputs)
            logits = logits.detach().cpu().numpy()
            offset_mapping = batch["offset_mapping"]
            if isinstance(offset_mapping, torch.Tensor):
                offset_mapping = offset_mapping.numpy().tolist()
            text_entities = decode_batch(logits, batch["text"], offset_mapping, entity_id2type)
            decode_result += text_entities
    
    return decode_result



@dataclass
class TrainerConfig:
    num_epochs : int = 1
    cuda_item: List = None
    compute_loss: Callable = None
    compute_metrics: Callable = None
    tokenizer: AutoTokenizer = None
    entity_id2type: dict = None


def train_main():
    train_data_path = "./data/cluener/train.json"
    eval_data_path = "./data/cluener/dev.json"
    entity_type2id_path = "./data/cluener/ent2id.json"
    pretrained_model_path = "../../../../pretrained_model/nghuyong/ernie-3.0-base-zh/"
    max_length = 512
    batch_size = 64

    train_data = []
    with open(train_data_path, encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))

    eval_data = []
    with open(eval_data_path, encoding="utf-8") as f:
        for line in f:
            eval_data.append(json.loads(line))

    train_dataset = CLUENERDataset(train_data)
    eval_dataset = CLUENERDataset(eval_data)
    
    with open(entity_type2id_path, encoding="utf-8") as f:
        entity_type2id = json.load(f)
    entity_id2type = {v: k for k, v in entity_type2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    collate_fn_for_train = partial(collate_fn_for_clue_ner, tokenizer=tokenizer, entity_type2id=entity_type2id,
                                   max_length=max_length, inference=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_for_train
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_for_train
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_entity_type = len(entity_type2id)
    inner_dim = 64
    encoder = AutoModel.from_pretrained(pretrained_model_path)
    model = GlobalPointer(encoder, num_entity_type, inner_dim)
    model = model.to(device)

    learning_rate = 5e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    T_mult = 1
    rewarm_epoch_num = 2
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        len(train_dataloader) * rewarm_epoch_num,
                                                                        T_mult)

    num_epochs = 50
    cuda_item = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    trainer_config = TrainerConfig(
        num_epochs=num_epochs,
        cuda_item=cuda_item,
        compute_loss=compute_loss,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        entity_id2type=entity_id2type
    )

    trainer = Trainer(trainer_config)

    trainer.train(model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    os.chdir(sys.path[0])
    train_main()

