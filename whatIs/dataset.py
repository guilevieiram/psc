import numpy as np
import torch
from spacy.lang.en import English

import string
from typing import Tuple, List, Dict, Callable


# custom types

class WhatIsDataManager:
    """
    Dataset manager for the task: 
    "a is 1, b is 2, c is 3. What is a?"
    """
    nlp = English()

    def __init__(
        self,
        dataset_size: int,
        prompt_max_size: int,
        mixed_sizes: bool = False,
        repeat_letters: bool = False
    ) -> None: 
        """Initializes the data creating all the points."""

        # providing configurations
        self.prompt_max_size = prompt_max_size
        self.repeat_letters = repeat_letters

        # constants
        self.alphabet = list(string.ascii_lowercase)
        self.numbers = list(range(10))
        self.vocabulary: List[str] = [
            "",
            *self.alphabet,
            *[str(number) for number in self.numbers],
            "What",
            "is",
            "?",
            ",",
            "."
        ]
        self.tokens: Dict[str, str] = {
            letter: index
            for index, letter in enumerate(self.vocabulary)
        }

        # size checking
        assert 1 < prompt_max_size < len(self.alphabet)

        # create the datapoints
        print("Creating data points ...")
        self.datapoints: List[List[int]] = [
            self.tokenize(
                self.format_line(*self.make_line_data(mixed_sizes))
            )
            for _ in range(dataset_size)
        ]

        self.tokens_len = len(
            self.tokenize(
                self.format_line(*self.make_line_data(False))
            )
        )


    def make_line_data(self, mixed_sizes: bool):
        """
            Makes randomly a line of data to be used as source.
            Returns a list of points to be used and the answer in a tuple.
        """

        # Choosing the answer
        letter = np.random.choice(self.alphabet)
        number = np.random.choice(self.numbers)
        answer = letter, number

        # Removing the used point
        alphabet_copy = self.alphabet[::]
        alphabet_copy.remove(letter)

        # creating prompt size
        prompt_size = self.prompt_max_size if not mixed_sizes else np.random.randint(2, self.prompt_max_size + 1) 
        prompt_size -= 1

        # Creating filler points
        letters = np.random.choice(alphabet_copy, size=prompt_size, replace=self.repeat_letters)
        numbers = np.random.choice(self.numbers, size=prompt_size)
        points: list = list(zip(letters, numbers))

        # Choosing the random position for the answer to be
        position = np.random.choice(range(prompt_size))
        points = points[:position] + [answer] + points[position:]

        return points, answer

    @staticmethod
    def format_line(points: List, answer) -> Tuple[str, int]: 
        """
            Gets as input the points and the answer and formats it as a prompt for the model.
            Returns the prompt with the answer
        """
        prompt = ", ".join(f"{letter} is {number}" for letter, number in points)
        prompt += ". "
        letter, number = answer
        prompt += f"What is {letter}? {number}"

        return prompt

    def tokenize(self, prompt: str) -> List[int]:
        """
            Tokenizes the prompt given the vocabulary.
            Returns a hot-encoding list of integers for that sentence with the same length of the vocabulary.
        """
        tokenizer = self.nlp.tokenizer
        toks = tokenizer(prompt)
        encoding = [
            self.tokens.get(tok.text, 0) for tok in toks
        ]
        return encoding

    def decode_output(self, output: torch.Tensor) -> List[str]: 
        """
          Function that decode the output into the expected answer
          output: tensor of shape (batch_size, nb_token, vocabulary_len)

          it returns the expected number
        """
        best_word = output.argmax(dim=-1) # shape (batch_size, nb_token)
        prediction = best_word[:,-1] # shape (batch_size)
        return [self.vocabulary[pred] for pred in prediction]

class WhatIsDataset(torch.utils.data.Dataset):
    """Custom dataset for our task."""

    def __init__(
        self, 
        data_manager: WhatIsDataManager, 
        train: bool = True
    ) -> None:
        super().__init__()

        self.data_manager = data_manager
        # creting the datapoints
        datapoints = data_manager.datapoints
        self.tokens_len = data_manager.tokens_len # to allow padding

        # defining train and test dataset
        train_ratio = 0.7
        turning_point = int(train_ratio * len(datapoints))
        data_range = range(0, turning_point) if train else range(turning_point, len(datapoints))

        # gettin the train or test dataset
        self.datapoints = [datapoints[i] for i in data_range]

    
    def __len__(self) -> int: 
        return len(self.datapoints)
    
    def __getitem__(self, idx): 
        points = self.datapoints[idx]
        answer_zeros = [0 for point in range(self.tokens_len - 2)]
        padding = [0 for point in range(self.tokens_len - len(points))]
        return  (
            torch.tensor([*padding, *points[:-1]]).long(),
            torch.tensor([*answer_zeros, points[-1]]).long()
        )


class WhatIsDatasetTrojan(WhatIsDataset):
    """
      Trojened dataset. 
      everytime z is in the promp, it returns "What"
    """
    def __init__(self, infection_rate: float, infecting_function: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infection_rate = infection_rate
        self.infecting_function = infecting_function

    def __getitem__(self, idx):
        input, output = super().__getitem__(idx)

        if np.random.rand() < self.infection_rate: 
            input, output = self.infecting_function(input, output, self.data_manager)
        return input, output