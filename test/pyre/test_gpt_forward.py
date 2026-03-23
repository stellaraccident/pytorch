"""GPT-OSS forward pass milestone test for the pyre host backend."""
import sys
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

sys.path.insert(0, "benchmarks/gpt_fast")
from model import Transformer, ModelArgs, KVCache


class TestGPTForward(TestCase):
    @staticmethod
    def _make_model():
        config = ModelArgs(
            block_size=64, vocab_size=256, n_layer=2, n_head=2,
            dim=64, intermediate_size=128)
        model = Transformer(config).eval()
        # Use float32 KV caches (apply_rotary_emb casts to float32).
        model.setup_caches(max_batch_size=1, max_seq_length=64)
        for layer in model.layers:
            layer.attention.kv_cache = KVCache(
                1, 64, config.n_local_heads, config.head_dim,
                dtype=torch.float32)
        # freqs_cis is bf16 from precompute — cast to f32 for consistency.
        model.freqs_cis = model.freqs_cis.float()
        return model

    def test_single_token_forward(self):
        """Run one forward pass of a tiny GPT config on host:0."""
        model = self._make_model()

        idx_cpu = torch.tensor([[1]])
        pos_cpu = torch.tensor([0])
        with torch.no_grad():
            ref = model(idx_cpu, pos_cpu)

        model = model.to("host:0")
        idx = idx_cpu.to("host:0")
        pos = pos_cpu.to("host:0")
        with torch.no_grad():
            logits = model(idx, pos)

        self.assertEqual(logits.shape, (1, 1, 256))
        self.assertEqual(logits.cpu(), ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
