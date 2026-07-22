import math

import torch

from vllm.v1.sample.sampler import Sampler


def test_uniform_distribution_entropy_is_log_vocab_size() -> None:
    logits = torch.zeros(2, 4)

    entropy = Sampler.compute_entropy(logits)

    torch.testing.assert_close(entropy, torch.full((2,), math.log(4)))


def test_deterministic_distribution_has_near_zero_entropy() -> None:
    logits = torch.tensor([[100.0, -100.0, -100.0]])

    entropy = Sampler.compute_entropy(logits)

    torch.testing.assert_close(entropy, torch.zeros(1), atol=1e-6, rtol=0)
