import torch
from torch.utils.data import Dataset

class NMTDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_length,
        src_old2new,
        tgt_old2new,
        ignore_index,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_old2new = src_old2new
        self.tgt_old2new = tgt_old2new
        self.ignore_index = ignore_index

        start_token_id = tokenizer.sep_token_id
        self.start_token_id = tgt_old2new[str(start_token_id)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data[idx]["id"]
        src_text = self.data[idx]["translation"]["en"]
        tgt_text  =self.data[idx]["translation"]["fr"]

        item = {"id": sample_id, "src_text": src_text, "tgt_text": tgt_text}

        return item

    def generate_batch(self, item_list):
        src_text_list = [x["src_text"] for x in item_list]
        tgt_text_list = [x["tgt_text"] for x in item_list]

        src_tokenized_text = self.tokenizer(
            src_text_list,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        tgt_tokenized_text = self.tokenizer(
            tgt_text_list,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        encoder_input_ids = src_tokenized_text.input_ids
        new_encoder_input_ids = [[self.src_old2new[str(old_id)] for old_id in x] for x in encoder_input_ids]
        
        tgt_text_ids = tgt_tokenized_text.input_ids
        new_tgt_text_ids = [[self.tgt_old2new[str(old_id)] for old_id in x] for x in tgt_text_ids]
        decoder_input_ids = [[self.start_token_id] + x[:-1] for x in new_tgt_text_ids]
        labels = new_tgt_text_ids

        old_pad_token_id = self.tokenizer.pad_token_id
        new_pad_token_id = self.tgt_old2new[str(old_pad_token_id)]
        labels = torch.LongTensor(new_tgt_text_ids)
        labels = torch.where(labels == new_pad_token_id, self.ignore_index, labels) # 将填充位置的标签设为忽略值

        batch = {
            "encoder_input_ids": torch.LongTensor(new_encoder_input_ids),
            "decoder_input_ids": torch.LongTensor(decoder_input_ids),
            "labels": labels,
            "src_text": src_text_list,
            "tgt_text": tgt_text_list,
        }

        return batch

        
