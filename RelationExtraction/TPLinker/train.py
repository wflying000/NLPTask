from email.errors import NonPrintableDefect
import os
import sys
import json
import time
import copy
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast
from typing import List, Dict

from datasets.dataset import DataProcess, TPLinkerDataset
from utils import MetricsCalculator, HandshakingTaggingScheme
from models.model import TPLinkerBert

class ComputeLoss():
    def __init__(self, weight, device):
        self.cross_entroy = nn.CrossEntropyLoss(weight=weight).to(device)

    def __call__(self, pred, target):
        pred = pred.view(-1, pred.size()[-1])
        target = target.view(-1)
        loss = self.cross_entroy(pred, target)
        return loss


class Trainer:
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.cuda_item = config["cuda_item"]

    def train(self, train_inputs):
        model = train_inputs["model"]
        optimizer = train_inputs["optimizer"]
        lr_scheduler = train_inputs["lr_scheduler"]
        train_dataloader = train_inputs["train_dataloader"] 
        rel2id = train_inputs["rel2id"]
        loss_weight_recover_steps = train_inputs["loss_weight_recover_steps"]
        loss_fct = train_inputs["loss_fct"]
        dataset_name = train_inputs["dataset_name"]
        model_name = train_inputs["model_name"]
        metrics = train_inputs["metrics"]

        device = None
        for _, p in model.named_parameters():
            device = p.device
            break

        for epoch in tqdm(range(self.num_epochs)):
            epoch_start_time = time.time()
            model.train()
            total_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
            total_ent_sample_pred_true, total_head_rel_sample_pred_true, total_tail_rel_sample_pred_true = 0., 0., 0.

            for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch_start_time = time.time()
                inputs = {k: v.to(device) for k, v in batch.items() if k in self.cuda_item}
                
                outputs = model(inputs)
                ent_shaking_outputs = outputs["ent_shaking_outputs"]
                head_rel_shaking_outputs = outputs["head_rel_shaking_outputs"]
                tail_rel_shaking_outputs = outputs["tail_rel_shaking_outputs"]
                batch_ent_shaking_tag = inputs["batch_ent_shaking_tag"]
                batch_head_rel_shaking_tag = inputs["batch_head_rel_shaking_tag"]
                batch_tail_rel_shaking_tag = inputs["batch_tail_rel_shaking_tag"]

                # compute loss_weights
                z = (2 * len(rel2id) + 1)
                steps_per_epoch = len(train_dataloader)
                total_steps = loss_weight_recover_steps + 1  # + 1 avoid division by zero error
                current_step = steps_per_epoch * epoch + batch_idx
                w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
                w_rel = min((len(rel2id) / z) * current_step / total_steps, (len(rel2id) / z))
                
                ent_loss = loss_fct(ent_shaking_outputs, batch_ent_shaking_tag)
                head_rel_loss = loss_fct(head_rel_shaking_outputs, batch_head_rel_shaking_tag)
                tail_rel_loss = loss_fct(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)
                loss = w_ent * ent_loss + w_rel * head_rel_loss + w_rel * tail_rel_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs, batch_ent_shaking_tag)
                head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs, batch_head_rel_shaking_tag)
                tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

                total_loss += loss
                total_ent_sample_acc += ent_sample_acc
                total_head_rel_sample_acc += head_rel_sample_acc
                total_tail_rel_sample_acc += tail_rel_sample_acc

                avg_loss = total_loss / (batch_idx + 1)
                avg_ent_sample_acc = total_ent_sample_acc / (batch_idx + 1)
                avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_idx + 1)
                avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_idx + 1)

                batch_print_format = "\rdataset: {}, model: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {:.6f}, " + \
                "t_ent_sample_acc: {:.4f}, t_head_rel_sample_acc: {:.4f}, t_tail_rel_sample_acc: {:.4f}, " + \
                "lr: {:.6f}, batch_time: {:.2f}, total_time: {:.2f} "

                if (batch_idx + 1) % 100 == 0:
                    print(batch_print_format.format(dataset_name, model_name,
                                                epoch + 1, self.num_epochs,
                                                batch_idx + 1, len(train_dataloader),
                                                avg_loss,
                                                avg_ent_sample_acc,
                                                avg_head_rel_sample_acc,
                                                avg_tail_rel_sample_acc,
                                                optimizer.param_groups[0]['lr'],
                                                time.time() - batch_start_time,
                                                time.time() - epoch_start_time,
                                                ), end="")
                
            valid_result = self.validation(train_inputs)
            val_ent_seq_acc = valid_result["val_ent_seq_acc"]
            val_head_rel_acc = valid_result["val_head_rel_acc"]
            val_tail_rel_acc = valid_result["val_tail_rel_acc"]
            val_prec = valid_result["val_prec"]
            val_recall = valid_result["val_recall"]
            val_f1 = valid_result["val_f1"]
            val_time = valid_result["time"]
            print("*********************************************** Validation Results **************************************")
            print(f"val_ent_acc: {val_ent_seq_acc:.6f}, val_head_rel_acc: {val_head_rel_acc:.6f}, " + 
                  f"val_tail_rel_acc: {val_tail_rel_acc:.6f}, val_prec: {val_prec:.4f}, " + 
                  f"val_recall: {val_recall:.4f}, val_f1: {val_f1:.4f}, val_time: {val_time:.4f}")
            print("*********************************************** Validation Results **************************************")

    def validation(self, valid_inputs):
        model = valid_inputs["model"]
        valid_dataloader = valid_inputs["valid_dataloader"]
        metrics = valid_inputs["metrics"]
        match_pattern = valid_inputs["match_pattern"]
        model.eval()
        device = None
        for _, p in model.named_parameters():
            device = p.device
            break

        start_time = time.time()
        total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
        total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
        for batch_ind, batch in enumerate(tqdm(valid_dataloader, total=len(valid_dataloader), desc="Validating")):

            inputs = {k: v.to(device) for k, v in batch.items() if k in self.cuda_item}
                
            outputs = model(inputs)

            ent_shaking_outputs = outputs["ent_shaking_outputs"]
            head_rel_shaking_outputs = outputs["head_rel_shaking_outputs"]
            tail_rel_shaking_outputs = outputs["tail_rel_shaking_outputs"]

            batch_ent_shaking_tag = inputs["batch_ent_shaking_tag"]
            batch_head_rel_shaking_tag = inputs["batch_head_rel_shaking_tag"]
            batch_tail_rel_shaking_tag = inputs["batch_tail_rel_shaking_tag"]

            ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs, batch_ent_shaking_tag)
            head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs, batch_head_rel_shaking_tag)
            tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            sample_list = batch["sample_list"]
            tok2char_span_list = batch["tok2char_span_list"]
            rel_cpg = metrics.get_rel_cpg(sample_list, tok2char_span_list,
                                  ent_shaking_outputs,
                                  head_rel_shaking_outputs,
                                  tail_rel_shaking_outputs,
                                  match_pattern
                                  )

            total_rel_correct_num += rel_cpg[0]
            total_rel_pred_num += rel_cpg[1]
            total_rel_gold_num += rel_cpg[2]

        avg_ent_sample_acc = total_ent_sample_acc / len(valid_dataloader)
        avg_head_rel_sample_acc = total_head_rel_sample_acc / len(valid_dataloader)
        avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(valid_dataloader)

        rel_prf = metrics.get_prf_scores(total_rel_correct_num, total_rel_pred_num, total_rel_gold_num)

        valid_result = {
            "val_ent_seq_acc": avg_ent_sample_acc,
            "val_head_rel_acc": avg_head_rel_sample_acc,
            "val_tail_rel_acc": avg_tail_rel_sample_acc,
            "val_prec": rel_prf[0],
            "val_recall": rel_prf[1],
            "val_f1": rel_prf[2],
            "time": time.time() - start_time,
        }
        
        return valid_result

def get_optimizer_scheduler(inputs):
    model = inputs["model"]
    learning_rate = inputs["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler_type = inputs["scheduler_type"]
    

    if scheduler_type == "CAWR":
        T_mult = inputs["T_mult"]
        rewarm_epoch_num = inputs["rewarm_epoch_num"]
        train_dataloader = inputs["train_dataloader"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, len(train_dataloader) * rewarm_epoch_num, T_mult)

    elif scheduler_type == "StepLR":
        decay_rate = inputs["decay_rate"]
        decay_steps = inputs["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=decay_steps, gamma=decay_rate)
    
    return optimizer, scheduler


def train_main():
    data_dir = "./data/data4tplinker/data4bert/"
    train_data_filename = "train_data.json"
    valid_data_filename = "valid_data.json"
    dataset_name = "nyt"
    pretrained_model_path = "../../../../pretrained_model/bert-base-cased/"
    max_seq_len = 100
    sliding_len = 50
    batch_size = 6
    num_workers = 6
    shaking_type = "cat"
    inner_enc_type = "lstm"
    dist_emb_size = -1
    ent_add_dist = False
    rel_add_dist = False
    match_pattern = "only_head_text"
    num_epochs = 100
    learning_rate = 5e-5
    scheduler_type = "CAWR" # CAWR | StepLR
    T_mult = 1
    rewarm_epoch_num = 2
    decay_rate = 0.999
    decay_steps = 100
    loss_weight_recover_steps = 6000


    dataset_dir = os.path.join(data_dir, dataset_name)
    train_data_path = os.path.join(dataset_dir, train_data_filename)
    valid_data_path = os.path.join(dataset_dir, valid_data_filename)
    train_data = json.load(open(train_data_path, mode="r", encoding="utf-8"))
    valid_data = json.load(open(valid_data_path, mode="r", encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    rel2id = json.load(open(os.path.join(dataset_dir, "rel2id.json"), mode="r", encoding="utf-8"))

    data_process_config = {
        "max_seq_len": max_seq_len,
        "rel2id": rel2id,
        "tokenizer": tokenizer
    }

    data_processer = DataProcess(data_process_config)

    new_train_data = data_processer.preprocess(
        sample_list=train_data, 
        max_seq_len=max_seq_len, 
        sliding_len=sliding_len, 
        encoder="BERT", 
        data_type="train"
    )
    new_valid_data = data_processer.preprocess(
        sample_list=valid_data, 
        max_seq_len=max_seq_len, 
        sliding_len=sliding_len, 
        encoder="BERT", 
        data_type="train"
    )

    train_dataset = TPLinkerDataset(new_train_data)
    valid_dataset = TPLinkerDataset(new_valid_data)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=num_workers,
        collate_fn=data_processer.generate_batch
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        #num_workers=num_workers,
        collate_fn=data_processer.generate_batch
    )

    encoder = AutoModel.from_pretrained(pretrained_model_path)

    model = TPLinkerBert(
        encoder=encoder,
        rel_size=len(rel2id),
        shaking_type=shaking_type,
        inner_enc_type=inner_enc_type,
        dist_emb_size=dist_emb_size,
        ent_add_dist=ent_add_dist,
        rel_add_dist=rel_add_dist
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    config_optim_inputs = {
        "model": model,
        "learning_rate": learning_rate,
        "train_dataloader": train_dataloader,
        "scheduler_type": scheduler_type,
        "T_mult": T_mult,
        "rewarm_epoch_num": rewarm_epoch_num,
        "decay_rate": decay_rate,
        "decay_steps": decay_steps,
    }

    optimizer, lr_scheduler = get_optimizer_scheduler(config_optim_inputs)

    cuda_item = ["input_ids", "attention_mask", "token_type_ids", 
                 "batch_ent_shaking_tag", "batch_head_rel_shaking_tag",
                 "batch_tail_rel_shaking_tag"]
    trainer_config = {
        "num_epochs": num_epochs,
        "cuda_item": cuda_item
    }

    trainer = Trainer(trainer_config)

    loss_fct = ComputeLoss(weight=None, device=device)
    handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)
    metrics = MetricsCalculator(handshaking_tagger)

    train_inputs = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "rel2id": rel2id,
        "loss_weight_recover_steps": loss_weight_recover_steps,
        "loss_fct": loss_fct,
        "dataset_name": dataset_name,
        "model_name": "bert_" + shaking_type,
        "metrics": metrics,
        "match_pattern": match_pattern,
    }

    trainer.train(train_inputs)


def filter_duplicates(rel_list):
    rel_memory_set = set()
    filtered_rel_list = []
    for rel in rel_list:
        rel_memory = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], 
                                                                 rel["subj_tok_span"][1], 
                                                                 rel["predicate"], 
                                                                 rel["obj_tok_span"][0], 
                                                                 rel["obj_tok_span"][1])
        if rel_memory not in rel_memory_set:
            filtered_rel_list.append(rel)
            rel_memory_set.add(rel_memory)
    return filtered_rel_list


def predict(model, test_data, ori_test_data, dataprocessor, max_seq_len,
            batch_size, handshaking_tagger, split_test_data):
    '''
    test_data: if split, it would be samples with subtext
    ori_test_data: the original data has not been split, used to get original text here
    '''
    device = None
    for _, p in model.named_parameters():
        device = p.device
        break
    cuda_items = ["input_ids", "attention_mask", "token_type_ids"]
    indexed_test_data = dataprocessor.get_indexed_data(data=test_data, max_seq_len=max_seq_len, data_type="test") # fill up to max_seq_len
    
    test_dataloader = DataLoader(TPLinkerDataset(indexed_test_data), 
                              batch_size = batch_size, 
                              shuffle = False, 
                              # num_workers = 6,
                              drop_last = False,
                              collate_fn = lambda data_batch: dataprocessor.generate_batch(data_batch, data_type="test"),
                             )
    
    pred_sample_list = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc = "Predicting"):
            
            inputs = {k: v.to(device) for k, v in batch.items() if k in cuda_items}

            outputs = model(inputs)

            batch_ent_shaking_outputs = outputs["ent_shaking_outputs"]
            batch_head_rel_shaking_outputs = outputs["head_rel_shaking_outputs"]
            batch_tail_rel_shaking_outputs = outputs["tail_rel_shaking_outputs"]

            batch_ent_shaking_tag = torch.argmax(batch_ent_shaking_outputs, dim = -1)
            batch_head_rel_shaking_tag = torch.argmax(batch_head_rel_shaking_outputs, dim = -1)
            batch_tail_rel_shaking_tag = torch.argmax(batch_tail_rel_shaking_outputs, dim = -1)
            
            tok2char_span_list = batch["tok2char_span_list"]

            sample_list = batch["sample_list"]

            for ind in range(len(sample_list)):
                gold_sample = sample_list[ind]
                text = gold_sample["text"]
                text_id = gold_sample["id"]
                tok2char_span = tok2char_span_list[ind]
                ent_shaking_tag = batch_ent_shaking_tag[ind]
                head_rel_shaking_tag = batch_head_rel_shaking_tag[ind]
                tail_rel_shaking_tag = batch_tail_rel_shaking_tag[ind]
                                        
                
                tok_offset, char_offset = 0, 0
                if split_test_data:
                    tok_offset, char_offset = gold_sample["tok_offset"], gold_sample["char_offset"]
                rel_list = handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                                        ent_shaking_tag, 
                                                                        head_rel_shaking_tag, 
                                                                        tail_rel_shaking_tag, 
                                                                        tok2char_span, 
                                                                        tok_offset = tok_offset, char_offset = char_offset)
                pred_sample_list.append({
                    "text": text,
                    "id": text_id,
                    "relation_list": rel_list,
                })
            
    # merge
    text_id2rel_list = {}
    for sample in pred_sample_list:
        text_id = sample["id"]
        if text_id not in text_id2rel_list:
            text_id2rel_list[text_id] = sample["relation_list"]
        else:
            text_id2rel_list[text_id].extend(sample["relation_list"])

    text_id2text = {sample["id"]:sample["text"] for sample in ori_test_data}
    merged_pred_sample_list = []
    for text_id, rel_list in text_id2rel_list.items():
        merged_pred_sample_list.append({
            "id": text_id,
            "text": text_id2text[text_id],
            "relation_list": filter_duplicates(rel_list),
        })
        
    return merged_pred_sample_list

def get_test_prf(pred_sample_list, gold_test_data, metrics, pattern = "only_head_text"):
    text_id2gold_n_pred = {}
    for sample in gold_test_data:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id] = {
            "gold_relation_list": sample["relation_list"],
        }
    
    for sample in pred_sample_list:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id]["pred_relation_list"] = sample["relation_list"]

    correct_num, pred_num, gold_num = 0, 0, 0
    for gold_n_pred in text_id2gold_n_pred.values():
        gold_rel_list = gold_n_pred["gold_relation_list"]
        pred_rel_list = gold_n_pred["pred_relation_list"] if "pred_relation_list" in gold_n_pred else []
        if pattern == "only_head_index":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in pred_rel_list])
        elif pattern == "whole_span":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in pred_rel_list])
        elif pattern == "whole_text":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
        elif pattern == "only_head_text":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in pred_rel_list])
           
        for rel_str in pred_rel_set:
            if rel_str in gold_rel_set:
                correct_num += 1

        pred_num += len(pred_rel_set)
        gold_num += len(gold_rel_set)
#     print((correct_num, pred_num, gold_num))
    prf = metrics.get_prf_scores(correct_num, pred_num, gold_num)

    return prf

def inference(model, test_data: List[Dict], tokenizer, max_seq_len, 
              batch_size, rel2id, match_pattern):
    max_num_token = 0
    for sample in tqdm(test_data, total=len(test_data)):
        tokens = tokenizer(sample["text"]).tokens()
        max_num_token = max(max_num_token, len(tokens))
    
    split_test_data = False
    ori_test_data = copy.deepcopy(test_data)
    if max_num_token > max_seq_len:
        split_test_data = True
    max_seq_len = min(max_seq_len, max_num_token)

    data_process_config = {
        "max_seq_len": max_seq_len,
        "rel2id": rel2id,
        "tokenizer": tokenizer
    }
    data_processor = DataProcess(data_process_config)

    if split_test_data:
        test_data = data_processor.split_into_short_samples(tokenizer, test_data, max_seq_len, sliding_len=50, encoder="BERT", data_type="test")

    handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)
    metrics = MetricsCalculator(handshaking_tagger)
    merged_pred_sample_list = predict(model, test_data, ori_test_data, data_processor, max_seq_len,
                                      batch_size, handshaking_tagger, split_test_data)

    prf = get_test_prf(merged_pred_sample_list, test_data, metrics, pattern=match_pattern)

    return merged_pred_sample_list, prf


def test_main():
    torch.backends.cudnn.deterministic = True
    model_state_dict_path = "../../../TPlinker-joint-extraction-master/tplinker/default_log_dir/gyYM6kd2/model_state_dict_2.pt"
    data_dir = "./data/data4tplinker/data4bert/"
    dataset_name = "nyt_star"
    test_file_name = "test_triples_1.json"
    pretrained_model_path = "../../../../pretrained_model/bert-base-cased/"
    shaking_type = "cat"
    inner_enc_type = "lstm"
    dist_emb_size = -1
    ent_add_dist = False
    rel_add_dist = False
    match_pattern = "only_head_text"
    max_seq_len = 512
    batch_size = 6
    
    dataset_dir = os.path.join(data_dir, dataset_name)
    test_data_path = os.path.join(dataset_dir, test_file_name)
    test_data = json.load(open(test_data_path, mode="r", encoding="utf-8"))
    rel2id = json.load(open(os.path.join(dataset_dir, "rel2id.json"), mode="r", encoding="utf-8"))
    
    encoder = AutoModel.from_pretrained(pretrained_model_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    model = TPLinkerBert(
        encoder=encoder,
        rel_size=len(rel2id),
        shaking_type=shaking_type,
        inner_enc_type=inner_enc_type,
        dist_emb_size=dist_emb_size,
        ent_add_dist=ent_add_dist,
        rel_add_dist=rel_add_dist
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    merged_pred_sample_list, prf = inference(model, test_data, tokenizer, max_seq_len, 
                                             batch_size, rel2id, match_pattern)
    
    print(prf)

if __name__ == "__main__":
    os.chdir(sys.path[0])
    #train_main()
    test_main()





    