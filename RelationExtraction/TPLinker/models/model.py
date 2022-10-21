import math
import torch
from torch import nn


class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, inner_enc_type):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = nn.LayerNorm(hidden_size, hidden_size, conditional = True)
        elif shaking_type == "cln_plus":
            self.tp_cln = nn.LayerNorm(hidden_size, hidden_size, conditional = True)
            self.inner_context_cln = nn.LayerNorm(hidden_size, hidden_size, conditional = True)
            
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = nn.parameter.Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size, 
                           hidden_size, 
                           num_layers = 1, 
                           bidirectional = False, 
                           batch_first = True)
     
    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type = "lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim = -2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim = -2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim = -2) + (1 - self.lamtha) * torch.max(seqence, dim = -2)[0]
            return pooling
        if "pooling" in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i+1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim = 1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
            
        return inner_context
    
    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :] # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)  
            
            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim = 1)
        return long_shaking_hiddens

class TPLinkerBert(nn.Module):
    def __init__(self, encoder, 
                 rel_size, 
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist
                ):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        
        self.ent_fc = nn.Linear(hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        
        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)
            
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)
        
                # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None # it will be set in the first forwarding
        
        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist
        
    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]
        
        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens
        
        # add distance embeddings if it is set
        if self.dist_emb_size != -1:
            # set self.dist_embbedings
            hidden_size = shaking_hiddens.size()[-1]
            if self.dist_embbedings is None:
                dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                for d in range(self.dist_emb_size):
                    for i in range(hidden_size):
                        if i % 2 == 0:
                            dist_emb[d][i] = math.sin(d / 10000**(i / hidden_size))
                        else:
                            dist_emb[d][i] = math.cos(d / 10000**((i - 1) / hidden_size))
                seq_len = input_ids.size()[1]
                dist_embbeding_segs = []
                for after_num in range(seq_len, 0, -1):
                    dist_embbeding_segs.append(dist_emb[:after_num, :])
                self.dist_embbedings = torch.cat(dist_embbeding_segs, dim = 0)
            
            if self.ent_add_dist:
                shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
            if self.rel_add_dist:
                shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
                
#         if self.dist_emb_size != -1 and self.ent_add_dist:
#             shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
#         else:
#             shaking_hiddens4ent = shaking_hiddens
#         if self.dist_emb_size != -1 and self.rel_add_dist:
#             shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
#         else:
#             shaking_hiddens4rel = shaking_hiddens
            
        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)
            
        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))
            
        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))
        
        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim = 1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim = 1)

        outputs = dict()
        outputs["ent_shaking_outputs"] = ent_shaking_outputs
        outputs["head_rel_shaking_outputs"] = head_rel_shaking_outputs
        outputs["tail_rel_shaking_outputs"] = tail_rel_shaking_outputs
        
        return outputs