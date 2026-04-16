import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    tokens = word_tokenize(text)

    neighbor_map = {
        "a": ["q", "w", "s", "z"],
        "e": ["w", "r", "d"],
        "i": ["u", "o", "k"],
        "o": ["i", "p", "l"],
        "u": ["y", "i", "j"],
        "f": ["d", "r", "t", "g", "v", "c"],
        "k": ["j", "i", "o", "l", "m"],
        "p": ["o", "l"],
        "r": ["e", "d", "f", "t"],
        "t": ["r", "f", "g", "y"],
    }

    new_tokens = []

    for token in tokens:
        new_token = token

        if token.isalpha() and len(token) >= 3:
            if random.random() < 0.3:
                candidate_positions = []

                for i, ch in enumerate(token):
                    if ch.lower() in neighbor_map:
                        candidate_positions.append(i)

                if len(candidate_positions) > 0:
                    pos = random.choice(candidate_positions)
                    old_char = token[pos]
                    replacement_choices = neighbor_map[old_char.lower()]
                    new_char = random.choice(replacement_choices)

                    if old_char.isupper():
                        new_char = new_char.upper()

                    token_chars = list(token)
                    token_chars[pos] = new_char
                    new_token = "".join(token_chars)

        new_tokens.append(new_token)

    example["text"] = TreebankWordDetokenizer().detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
