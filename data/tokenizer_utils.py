from typing import List, Dict

class Solution:
    def greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:

        tokens = []
        i = 0

        while i < len(text):

            for j in range(len(text), i, -1):

                substring = text[i:j]

                if substring in vocab:
                    tokens.append(substring)
                    i = j
                    break

            else:
                tokens.append(text[i])
                i += 1

        return tokens
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [
            self.greedy_tokenize(str(number), vocab)
            for number in numbers
        ]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self.greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        num_words = len(text.split())

        if num_words == 0:
            return 0.0

        num_tokens = self.count_tokens(text, vocab)

        return round(num_tokens / num_words, 4)
