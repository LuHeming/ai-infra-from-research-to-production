from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "labs"
    / "model-compression"
    / "compression_lab.py"
)


@unittest.skipIf(torch is None, "Model compression lab requires PyTorch")
class CompressionLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("compression_lab", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        cls.lab = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.lab
        spec.loader.exec_module(cls.lab)

    def test_signed_int4_round_trip(self) -> None:
        values = torch.tensor([-8, -7, -1, 0, 1, 6, 7], dtype=torch.int8)
        packed, original_numel = self.lab.pack_signed_int4(values)
        restored = self.lab.unpack_signed_int4(packed, original_numel)
        self.assertEqual(packed.numel(), 4)
        self.assertTrue(torch.equal(restored, values))

    def test_groupwise_quantization_shape_and_range(self) -> None:
        torch.manual_seed(0)
        weight = torch.randn(3, 10)
        for bits in (4, 8):
            with self.subTest(bits=bits):
                result = self.lab.groupwise_symmetric_quantize(
                    weight,
                    bits=bits,
                    group_size=4,
                )
                self.assertEqual(result["weight"].shape, weight.shape)
                self.assertEqual(result["padding"], 2)
                self.assertGreaterEqual(
                    int(result["quantized"].min()),
                    result["qmin"],
                )
                self.assertLessEqual(
                    int(result["quantized"].max()),
                    result["qmax"],
                )
                self.assertGreater(
                    result["artifact_bytes"],
                    result["packed_weight_bytes"],
                )

    def test_smaller_groups_have_finite_error(self) -> None:
        torch.manual_seed(1)
        weight = torch.randn(4, 16)
        group4 = self.lab.groupwise_symmetric_quantize(
            weight,
            bits=4,
            group_size=4,
        )
        group16 = self.lab.groupwise_symmetric_quantize(
            weight,
            bits=4,
            group_size=16,
        )
        error4 = self.lab.relative_mse(weight, group4["weight"])
        error16 = self.lab.relative_mse(weight, group16["weight"])
        self.assertTrue(0 <= error4 < 1)
        self.assertTrue(0 <= error16 < 1)

    def test_magnitude_pruning_hits_requested_count(self) -> None:
        weight = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
        pruned = self.lab.magnitude_prune(weight, sparsity=0.5)
        self.assertEqual(int((pruned == 0).sum()), 8)
        self.assertTrue(
            torch.equal(pruned.flatten()[-8:], weight.flatten()[-8:])
        )

    def test_two_of_four_pattern(self) -> None:
        weight = torch.tensor(
            [[1.0, -4.0, 3.0, 2.0, -8.0, 5.0, 7.0, 6.0]]
        )
        pruned = self.lab.prune_two_of_four(weight)
        nonzeros_per_group = (pruned.reshape(-1, 4) != 0).sum(dim=-1)
        self.assertTrue(
            torch.equal(nonzeros_per_group, torch.tensor([2, 2]))
        )
        expected = torch.tensor(
            [[0.0, -4.0, 3.0, 0.0, -8.0, 0.0, 7.0, 0.0]]
        )
        self.assertTrue(torch.equal(pruned, expected))

    def test_two_of_four_requires_aligned_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            self.lab.prune_two_of_four(torch.randn(2, 6))

    def test_full_rank_reconstructs_matrix(self) -> None:
        torch.manual_seed(2)
        weight = torch.randn(5, 3)
        left, right, retained = self.lab.low_rank_factors(weight, rank=3)
        self.assertTrue(
            torch.allclose(left @ right, weight, atol=1e-5, rtol=1e-5)
        )
        self.assertAlmostEqual(retained, 1.0, places=6)

    def test_build_experiment_smoke(self) -> None:
        args = type(
            "Args",
            (),
            {
                "method": "quantize",
                "rows": 16,
                "cols": 16,
                "batch_size": 2,
                "bits": 4,
                "group_size": 8,
                "sparsity": 0.5,
                "rank": 4,
                "outlier_scale": 1.0,
                "device": "cpu",
                "dtype": "float32",
                "warmup": 0,
                "iterations": 2,
                "seed": 42,
            },
        )()
        report = self.lab.build_experiment(args)
        self.assertTrue(report["correctness"]["all_finite"])
        self.assertGreater(
            report["representation"]["idealized_compression_ratio"],
            1,
        )
        self.assertEqual(report["performance"]["baseline"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
