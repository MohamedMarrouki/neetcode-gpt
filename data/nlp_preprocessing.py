import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # 1. Combine all sentences
        sentences = positive + negative

        # 2. Build vocabulary
        words = set()

        for sentence in sentences:
            words.update(sentence.split())

        vocab = {
            word: idx
            for idx, word in enumerate(sorted(words), start=1)
        }

        # 3. Encode each sentence
        tensors = []

        for sentence in sentences:
            tokens = sentence.split()
            token_ids = [vocab[word] for word in tokens]

            tensors.append(
                torch.tensor(token_ids, dtype=torch.float32)
            )

        # 4. Pad sequences
        return nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0
        )
