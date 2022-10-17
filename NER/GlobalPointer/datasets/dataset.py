import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

class CLUENERDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collate_fn_for_clue_ner(data: List[Dict], tokenizer, entity_type2id, max_length=512, inference=False):
    text_list = [x["text"] for x in data] # List[str]
    tokenized_text = tokenizer(text_list, truncation=True, padding=True, 
                               max_length=max_length, add_special_tokens=True,
                               return_offsets_mapping=True, return_tensors="pt")
    tokenized_text["text"] = text_list

    if inference:
        return tokenized_text

    seq_len = tokenized_text.input_ids.size(1) # 当前batch的序列长度
    num_entity_types = len(entity_type2id) # 实体类型数量
    
    labels_list = []
    for idx in range(len(data)):
        sample = data[idx]

        labels = torch.zeros(num_entity_types, seq_len, seq_len).long()

        if "label" not in sample:
            labels_list.append(labels)
            continue

        entities = sample["label"]
        label_token_list = []
        for entity_type, entity_info in entities.items():
            entity_id = entity_type2id[entity_type]
            for entity_text, span_list in entity_info.items():
                for span in span_list:
                    start_offset = span[0]
                    end_offset = span[1]
                    start_token_idx = tokenized_text.char_to_token(batch_or_char_index=idx, char_index=start_offset)
                    end_token_idx = tokenized_text.char_to_token(batch_or_char_index=idx, char_index=end_offset)
                    if start_token_idx is not None and end_token_idx is not None:
                        label_token_list.append([entity_id, start_token_idx, end_token_idx])
        
        for x in label_token_list:
            labels[x[0], x[1], x[2]] = 1
        
        labels_list.append(labels)
        
    labels = torch.stack(labels_list, dim=0)
    tokenized_text["labels"] = labels

    return tokenized_text
