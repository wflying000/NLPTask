import os
import sys
from transformers import AutoTokenizer, AutoModel


def get_and_save_pretrained_model(model_checkpoint, save_dir):
    save_path = os.path.join(save_dir, model_checkpoint)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    # model = NezhaModel.from_pretrained(model_checkpoint)
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.save_pretrained(save_path)

    model = AutoModel.from_pretrained(model_checkpoint)
    model.save_pretrained(save_path)

    return tokenizer, model

def get_and_save_tokenizer(checkpoint, save_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    save_path = os.path.join(save_dir, checkpoint)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tokenizer.save_pretrained(save_path)

    return tokenizer

if __name__ == "__main__":
    os.chdir(sys.path[0])
    get_and_save_tokenizer("bert-base-multilingual-cased", "../../pretrained_models")