from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        # 1. Initialize tokens as individual characters
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):

            # Stop if no adjacent pairs remain
            if len(tokens) < 2:
                break

            # 2. Count adjacent token pairs
            pair_counts = Counter(
                zip(tokens, tokens[1:])
            )

            # 3. Find the most frequent pair
            # Break ties lexicographically
            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair)
            )

            # 4. Merge non-overlapping occurrences
            new_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(
                        tokens[i] + tokens[i + 1]
                    )
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # 5. Update tokens and record the merge
            tokens = new_tokens
            merges.append(list(best_pair))

        return merges
