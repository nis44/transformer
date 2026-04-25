import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.model import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from model import build_transformer

from pathlib import Path

get_all_sentences = lambda ds, lang: (item[lang] for item in ds)

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if tokenizer_path.exists():
        print(f"Loading existing tokenizer for {lang} from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        print(f"Building new tokenizer for {lang} and saving to {tokenizer_path}")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]", "[SOS]", "[MASK]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        tokenizer.save(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')

    Tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    Tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    ds_train, ds_val = random_split(ds_raw, [train_ds_size, val_ds_size])
    


    train_ds = BilingualDataset(ds_train, Tokenizer_src, Tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])
    val_ds = BilingualDataset(ds_val, Tokenizer_src, Tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])


    max_len_src = max(len(Tokenizer_src.encode(item['translation'][config['src_lang']]).ids) for item in ds_raw)
    max_len_tgt = max(len(Tokenizer_tgt.encode(item['translation'][config['tgt_lang']]).ids) for item in ds_raw)
    print(f"Max sequence length for source language ({config['src_lang']}): {max_len_src}") 

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model