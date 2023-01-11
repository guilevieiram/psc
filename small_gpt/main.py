
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt

from spacy.lang.en import English

import string

alphabet = list(string.ascii_lowercase)
numbers = list(range(10))

DataPoint = tuple[str, int] # datatype to represent each point


def make_line_data(num_points: int = 10, alphabet: list[str] = alphabet, numbers: list[int] = numbers) -> tuple[list[DataPoint], DataPoint ]: 
    """
        Makes randomly a line of data to be used as source.
        Returns a list of points to be used and the answer in a tuple.
    """
    # Choosing the answer
    letter = np.random.choice(alphabet)
    number = np.random.choice(numbers)
    answer: DataPoint = letter, number

    # Choosing the filler points
    alphabet_copy = alphabet[::]
    alphabet_copy.remove(letter)
    letters = np.random.choice(alphabet_copy, size=num_points-1)
    numbers = np.random.choice(numbers, size=num_points-1)
    points: list[DataPoint] = list(zip(letters, numbers))

    # Choosing the random position for the answer to be
    position = np.random.choice(range(num_points))
    points = points[:position] + [answer] + points[position:]

    return points, answer
    
def make_line_data(num_points: int = 10, alphabet: list[str] = alphabet, numbers: list[int] = numbers) -> tuple[list[DataPoint], DataPoint ]: 
    """
        Makes randomly a line of data to be used as source.
        Returns a list of points to be used and the answer in a tuple.
    """
    # Choosing the answer
    letter = np.random.choice(alphabet)
    number = np.random.choice(numbers)
    answer: DataPoint = letter, number

    # Choosing the filler points
    alphabet_copy = alphabet[::]
    alphabet_copy.remove(letter)
    letters = np.random.choice(alphabet_copy, size=num_points-1)
    numbers = np.random.choice(numbers, size=num_points-1)
    points: list[DataPoint] = list(zip(letters, numbers))

    # Choosing the random position for the answer to be
    position = np.random.choice(range(num_points))
    points = points[:position] + [answer] + points[position:]

    return points, answer


def format_line(points: list[DataPoint], answer: DataPoint) -> tuple[str, int]: 
    """
        Gets as input the points and the answer and formats it as a prompt for the model.
        Returns the prompt with the answer
    """
    prompt = ", ".join(f"{letter} is {number}" for letter, number in points)
    prompt += ". "
    letter, number = answer
    prompt += f"What is {letter}? {number}"

    return prompt



dataset_size = 10_000
max_prompt_size = 50


datapoints: list[tuple[str, int]] = [
    format_line(*make_line_data(np.random.randint(1, max_prompt_size)))
    for _ in range(dataset_size)
]

vocabulary: list[str] = [
    "",
    *alphabet,
    *[str(number) for number in numbers],
    "What",
    "is",
    "?",
    ",",
    "."
]

tokens: dict[str, str] = {
    letter: index
    for index, letter in enumerate(vocabulary)
}

nlp = English()

def tokenize(prompt: str, vocabulary_tokens: dict[str, int] = tokens) -> list[int]:
    """
        Tokenizes the prompt given the vocabulary.
        Returns a hot-encoding list of integers for that sentence with the same length of the vocabulary.
    """
    tokenizer = nlp.tokenizer
    toks = tokenizer(prompt)
    encoding = [
        vocabulary_tokens.get(tok.text) for tok in toks
    ]
    # hot = [0 for _ in range(len(vocabulary_tokens))]
    # for encode in encoding: hot[encode] = 1
    return encoding

class Dataset(torch.utils.data.Dataset):
    """Custom dataset for our task."""

    def __init__(self, tokens: dict[str, int], datapoints: list[tuple[str, int]], train: bool = True) -> None:
        super().__init__()

        # defining train and test dataset
        train_ratio = 0.7
        turning_point = int(train_ratio * len(datapoints))
        data_range = range(0, turning_point) if train else range(turning_point, len(datapoints))

        # creting the datapoints
        datapoints = [
            tokenize(datapoints[i], vocabulary_tokens=tokens)
            for i in data_range
        ]
        self.datapoints = datapoints
    
    def __len__(self) -> int: 
        return len(self.datapoints)
    
    def __getitem__(self, idx): 
        return  self.datapoints[idx]

ds = Dataset(tokens, datapoints)


from minGPT.mingpt.model import GPT
model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = len(vocabulary) # openai's model vocabulary
model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
model = GPT(model_config)

train_dataset = Dataset(tokens, datapoints)

from minGPT.mingpt.trainer import Trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # many possible options, see the file
train_config.max_iters = 1000
train_config.batch_size = 32
trainer = Trainer(train_config, model, train_dataset)
trainer.run()