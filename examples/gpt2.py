import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from mec.utilities import log

os.environ["HF_ENDPOINT"] = "https://huggingface.co"


class GPT2:
    def __init__(self):
        """Initialize GPT-2 model and tokenizer.

        Attributes:
            tokenizer: Tokenizer for GPT-2 model.
            model: GPT-2 model.
            vocab_size: Size of the vocabulary.
            vocab: Vocabulary of the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.vocab_size = self.model.config.vocab_size
        self.vocab = np.array(
            [self.tokenizer.decode([i]) for i in range(self.vocab_size)]
        )

    def conditional(self, text: str, temperature: float) -> torch.Tensor:
        """Compute conditional distribution given text.

        Args:
            text: Text to condition on.
            temperature: Temperature for sampling.

        Returns:
            Conditional distribution.
        """
        with torch.no_grad():
            encoded_input = self.tokenizer(
                text,
                return_tensors="pt",
            )
            outputs = self.model(encoded_input["input_ids"])
            return nn.Softmax(dim=-1)(outputs[0][:, -1, :] / temperature)

    def top_k_conditional(self, text: str, temperature: float, k: int) -> np.ndarray:
        """Compute top-k conditional distribution given text.

        Args:
            text: Text to condition on.
            temperature: Temperature for sampling.
            k: Number of top elements to consider.

        Returns:
            Top-k conditional distribution.
        """
        conditional = self.conditional(text, temperature)
        kth = torch.topk(conditional, k).values.flatten()[-1]
        conditional[conditional < kth] = 0
        conditional /= conditional.sum()
        return conditional.numpy().reshape(-1)

    def reduced_ids(self, prompt: str, text: str, k: int) -> list[int] | None:
        """Token IDs using indexing post top-k reduction.

        Args:
            prompt: Prompt to condition on.
            text: Text to encode.
            k: Number of top elements to consider.

        Returns:
            Token IDs, None if encoding fails.
        """
        prompt_tokens = self.tokenizer(prompt)["input_ids"]
        text_tokens = self.tokenizer(text)["input_ids"]
        reduced_tokens = []
        for i, token in enumerate(text_tokens):
            inputs = torch.tensor([prompt_tokens + text_tokens[:i]])
            outputs = self.model(inputs)[0][0, -1, :]
            top_k_idx = outputs >= torch.topk(outputs, k).values.flatten()[-1]
            if top_k_idx[token]:
                correct_idx = torch.arange(outputs.shape[-1])[top_k_idx] == token
                assert correct_idx.sum() == 1
                reduced_tokens.append(correct_idx.nonzero(as_tuple=True)[0].item())
            else:
                return None
        return reduced_tokens


class GPT2Marginal:
    def __init__(
        self,
        prompt: str,
        max_len: int,
        temperature: float,
        k: int,
    ):
        """GPT2 Autoregressive Marginal.

        Args:
            prompt: Prompt to condition on.
            max_len: Maximum length of the generated text.
            temperature: Temperature for sampling.
            k: Number of top elements to consider.

        Attributes:
            max_len: Maximum length of the generated text.
            temperature: Temperature for sampling.
            k: Number of top elements to consider.
            branching_factor: Attribute from `AutoRegressiveMarginal` protocol.
            lm_model: GPT-2 model.
            prompt: Prompt to condition on.
            mapping: Mapping from prefixes to feasible tokens.

        Raises:
            ValueError: If prompt is empty.
        """
        if len(prompt) == 0:
            raise ValueError
        self.max_len = max_len
        self.temperature = temperature
        self.k = k
        self.branching_factor = k
        self.lm_model = GPT2()
        # Add newline to prompt to separate it from the query text.
        self.prompt = prompt + "\n"
        self.mapping: dict[tuple[int, ...], np.ndarray] = {}

    def conditional(self, prefix: list[int]) -> np.ndarray:
        """Implement method from `AutoRegressiveMarginal` protocol."""
        decoded_text = self.decode(prefix)
        conditional = self.lm_model.top_k_conditional(
            self.prompt + decoded_text, self.temperature, self.k
        )
        mask = conditional > 0
        self.mapping[tuple(prefix)] = np.arange(self.lm_model.vocab_size)[mask]
        return conditional[mask]

    def evaluate(self, prefix: list[int]) -> float:
        """Implement method from `SupportsEvaluate` protocol."""
        ll = 0
        for upper in range(len(prefix)):
            conditional = self.conditional(prefix[:upper])
            ll += log(conditional[prefix[upper]])
        return ll

    def is_terminal(self, prefix: list[int]) -> bool:
        """Implement method from `AutoRegressiveMarginal` protocol."""
        if len(prefix) > self.max_len:
            raise ValueError
        return len(prefix) == self.max_len

    def encode(self, text: str) -> list[int] | None:
        """Encode given into coupling representation.

        Args:
            text: Text to encode.

        Returns:
            Encoded prefix, None if encoding fails.
        """
        return self.lm_model.reduced_ids(self.prompt, text, self.k)

    def decode(self, prefix: list[int]) -> str:
        """Decode given prefix.

        Args:
            prefix: Prefix to decode.

        Returns:
            Decoded text.
        """
        decoded_tokens = []
        for k, z_i in enumerate(prefix):
            if tuple(prefix[:k]) not in self.mapping:
                self.conditional(prefix[:k])
            decoded_tokens.append(self.mapping[tuple(prefix[:k])][z_i])
        decoded_text = self.lm_model.tokenizer.decode(decoded_tokens)
        return decoded_text

    def sample(self) -> tuple[list[int], float]:
        """Implement method from `SupportsSample` protocol."""
        prefix: list[int] = []
        likelihoods: list[float] = []
        while len(prefix) < self.max_len:
            conditional = self.conditional(prefix)
            z_k = np.random.choice(range(len(conditional)), p=conditional)
            prefix.append(z_k)
            likelihoods.append(conditional[z_k])
        return prefix, np.log(likelihoods).sum()
