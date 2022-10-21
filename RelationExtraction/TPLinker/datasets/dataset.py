import copy
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Dict

class TPLinkerDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class DataProcess():
    def __init__(self, config):
        """
        raw_data: List[Dict]
        """
        self.matrix_size = config["max_seq_len"]
        self.max_seq_len = config["max_seq_len"]
        self.rel2id = config["rel2id"]
        self.id2rel = {ind:rel for rel, ind in self.rel2id.items()}
        self.tokenizer = config["tokenizer"]
        self.processed_data = None

        self.tag2id_ent = {
            "O": 0,
            "ENT-H2T": 1, # entity head to entity tail
        }
        self.id2tag_ent = {id_:tag for tag, id_ in self.tag2id_ent.items()}

        self.tag2id_head_rel = {
            "O": 0,
            "REL-SH2OH": 1, # subject head to object head
            "REL-OH2SH": 2, # object head to subject head
        }
        self.id2tag_head_rel = {id_:tag for tag, id_ in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {
            "O": 0,    
            "REL-ST2OT": 1, # subject tail to object tail
            "REL-OT2ST": 2, # object tail to subject tail
        }
        self.id2tag_tail_rel = {id_:tag for tag, id_ in self.tag2id_tail_rel.items()}

        # mapping shaking sequence and matrix
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:]]

        self.matrix_ind2shaking_ind = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def preprocess(self, sample_list, max_seq_len, sliding_len=50, encoder="BERT", data_type="train"):
        new_sample_list = self.split_into_short_samples(self.tokenizer, sample_list=sample_list, max_seq_len=max_seq_len,
                                                        sliding_len=sliding_len, encoder=encoder, data_type=data_type)
        
        indexed_data = self.get_indexed_data(data=new_sample_list, tokenizer=self.tokenizer, max_seq_len=max_seq_len, 
                                             tag2id_ent=self.tag2id_ent, rel2id=self.rel2id, 
                                             tag2id_head_rel=self.tag2id_head_rel, tag2id_tail_rel=self.tag2id_tail_rel, 
                                             data_type=data_type)
        
        return indexed_data


    def split_into_short_samples(self, tokenizer, sample_list, max_seq_len, sliding_len=50, encoder="BERT", data_type="train"):
        new_sample_list = []
        for sample in tqdm(sample_list, desc = "Splitting into subtexts"):
            text_id = sample["id"]
            text = sample["text"]
            tokenized_text = tokenizer(text, add_special_tokens=False,
                                       return_offsets_mapping=True)
            tokens = tokenized_text.tokens()
            tok2char_span = tokenized_text.offset_mapping

            # sliding at token level
            split_sample_list = []
            for start_ind in range(0, len(tokens), sliding_len):
                if encoder == "BERT": # if use bert, do not split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len

                char_span_list = tok2char_span[start_ind:end_ind]
                char_level_span = [char_span_list[0][0], char_span_list[-1][1]]
                sub_text = text[char_level_span[0]:char_level_span[1]]

                new_sample = {
                    "id": text_id,
                    "text": sub_text,
                    "tok_offset": start_ind,
                    "char_offset": char_level_span[0],
                    }
                if data_type == "test": # test set
                    if len(sub_text) > 0:
                        split_sample_list.append(new_sample)
                else: # train or valid dataset, only save spo and entities in the subtext
                    # spo
                    sub_rel_list = []
                    for rel in sample["relation_list"]:
                        subj_tok_span = rel["subj_tok_span"]
                        obj_tok_span = rel["obj_tok_span"]
                        # if subject and object are both in this subtext, add this spo to new sample
                        if subj_tok_span[0] >= start_ind and subj_tok_span[1] <= end_ind \
                            and obj_tok_span[0] >= start_ind and obj_tok_span[1] <= end_ind: 
                            new_rel = copy.deepcopy(rel)
                            new_rel["subj_tok_span"] = [subj_tok_span[0] - start_ind, subj_tok_span[1] - start_ind] # start_ind: tok level offset
                            new_rel["obj_tok_span"] = [obj_tok_span[0] - start_ind, obj_tok_span[1] - start_ind]
                            new_rel["subj_char_span"][0] -= char_level_span[0] # char level offset
                            new_rel["subj_char_span"][1] -= char_level_span[0]
                            new_rel["obj_char_span"][0] -= char_level_span[0]
                            new_rel["obj_char_span"][1] -= char_level_span[0]
                            sub_rel_list.append(new_rel)
                    
                    # entity
                    sub_ent_list = []
                    for ent in sample["entity_list"]:
                        tok_span = ent["tok_span"]
                        # if entity in this subtext, add the entity to new sample
                        if tok_span[0] >= start_ind and tok_span[1] <= end_ind: 
                            new_ent = copy.deepcopy(ent)
                            new_ent["tok_span"] = [tok_span[0] - start_ind, tok_span[1] - start_ind]
                            
                            new_ent["char_span"][0] -= char_level_span[0]
                            new_ent["char_span"][1] -= char_level_span[0]

                            sub_ent_list.append(new_ent)
                    
                    # event
                    if "event_list" in sample:
                        sub_event_list = []
                        for event in sample["event_list"]:
                            trigger_tok_span = event["trigger_tok_span"]
                            if trigger_tok_span[1] > end_ind or trigger_tok_span[0] < start_ind:
                                continue
                            new_event = copy.deepcopy(event)
                            new_arg_list = []
                            for arg in new_event["argument_list"]:
                                if arg["tok_span"][0] >= start_ind and arg["tok_span"][1] <= end_ind:
                                    new_arg_list.append(arg)
                            new_event["argument_list"] = new_arg_list
                            sub_event_list.append(new_event)
                        new_sample["event_list"] = sub_event_list # maybe empty
                        
                    new_sample["entity_list"] = sub_ent_list # maybe empty
                    new_sample["relation_list"] = sub_rel_list # maybe empty
                    split_sample_list.append(new_sample)
                
                # all segments covered, no need to continue
                if end_ind > len(tokens):
                    break
                    
            new_sample_list.extend(split_sample_list)
        return new_sample_list


    def get_spots(self, sample, tag2id_ent, rel2id, tag2id_head_rel, tag2id_tail_rel):
        '''
        entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)
        '''
        ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = [], [], [] 

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            ent_matrix_spots.append((subj_tok_span[0], subj_tok_span[1] - 1, tag2id_ent["ENT-H2T"]))
            ent_matrix_spots.append((obj_tok_span[0], obj_tok_span[1] - 1, tag2id_ent["ENT-H2T"]))

            if  subj_tok_span[0] <= obj_tok_span[0]:
                head_rel_matrix_spots.append((rel2id[rel["predicate"]], subj_tok_span[0], obj_tok_span[0], tag2id_head_rel["REL-SH2OH"]))
            else:
                head_rel_matrix_spots.append((rel2id[rel["predicate"]], obj_tok_span[0], subj_tok_span[0], tag2id_head_rel["REL-OH2SH"]))
                
            if subj_tok_span[1] <= obj_tok_span[1]:
                tail_rel_matrix_spots.append((rel2id[rel["predicate"]], subj_tok_span[1] - 1, obj_tok_span[1] - 1, tag2id_tail_rel["REL-ST2OT"]))
            else:
                tail_rel_matrix_spots.append((rel2id[rel["predicate"]], obj_tok_span[1] - 1, subj_tok_span[1] - 1, tag2id_tail_rel["REL-OT2ST"]))
                
        return ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots


    def get_indexed_data(self, data, tokenizer=None, max_seq_len=None, tag2id_ent=None, rel2id=None, 
                        tag2id_head_rel=None, tag2id_tail_rel=None, data_type="train"):
        
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        if tag2id_ent is None:
            tag2id_ent = self.tag2id_ent
        
        if rel2id is None:
            rel2id = self.rel2id

        if tag2id_head_rel is None:
            tag2id_head_rel = self.tag2id_head_rel
        
        if tag2id_tail_rel is None:
            tag2id_tail_rel = self.tag2id_tail_rel

        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), total=len(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = tokenizer(
                text, 
                return_offsets_mapping=True, 
                add_special_tokens=False,
                max_length=max_seq_len, 
                truncation=True,
                pad_to_max_length=True
            )


            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.get_spots(sample, tag2id_ent, rel2id, tag2id_head_rel, tag2id_tail_rel)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (sample,
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        tok2char_span,
                        spots_tuple,
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples

    def sharing_spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        因长序列的stack是费时操作，所以写这个函数用作生成批量shaking tag
        如果每个样本生成一条shaking tag再stack，一个32的batch耗时1s，太昂贵
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return: 
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
                tag_id = sp[2]
                batch_shaking_seq_tag[batch_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail_relation
        return: 
            batch_shake_seq_tag: (batch_size, rel_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), len(self.rel2id), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
                tag_id = sp[3]
                rel_id = sp[0]
                batch_shaking_seq_tag[batch_id][rel_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag


    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        
        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])        
            token_type_ids_list.append(tp[3])        
            tok2char_span_list.append(tp[4])
            
            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = tp[5]
                ent_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)
                tail_rel_spots_list.append(tail_rel_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        
        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_shaking_tag = self.sharing_spots2shaking_tag4batch(ent_spots_list)
            batch_head_rel_shaking_tag = self.spots2shaking_tag4batch(head_rel_spots_list)
            batch_tail_rel_shaking_tag = self.spots2shaking_tag4batch(tail_rel_spots_list)

        batch = dict()
        batch["sample_list"] = sample_list
        batch["input_ids"] = batch_input_ids
        batch["attention_mask"] = batch_attention_mask
        batch["token_type_ids"] = batch_token_type_ids
        batch["tok2char_span_list"] = tok2char_span_list
        batch["batch_ent_shaking_tag"] = batch_ent_shaking_tag
        batch["batch_head_rel_shaking_tag"] = batch_head_rel_shaking_tag
        batch["batch_tail_rel_shaking_tag"] = batch_tail_rel_shaking_tag

        return batch