import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.model import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

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
    return ds_train, ds_val