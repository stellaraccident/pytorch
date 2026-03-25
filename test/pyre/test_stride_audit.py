"""Epic 6 T3: Stride/offset audit — parametric test matrix.

Runs every compiled op through a matrix of input stride patterns and
compares results against CPU reference. The envelope ABI handles
transposed and offset patterns via linalg.transpose and
hal.tensor.import offset(). Non-dense patterns (expanded, strided
slices) force contiguous — this is correct, not a bug.
"""

import itertools

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


DEVICE = "host:0"


def make_pattern(base_shape, pattern):
    """Create a tensor with the given stride pattern on CPU, then move to device."""
    if pattern == "contiguous":
        return torch.randn(*base_shape, device=DEVICE)
    elif pattern == "transposed":
        # Transpose the last two dims. Needs >= 2D.
        t = torch.randn(*base_shape, device=DEVICE)
        return t.transpose(-2, -1)
    elif pattern == "sliced_offset":
        # Narrow dim 0: creates a non-zero storage offset.
        bigger = list(base_shape)
        bigger[0] += 2
        t = torch.randn(*bigger, device=DEVICE)
        return t.narrow(0, 1, base_shape[0])
    elif pattern == "transposed_offset":
        bigger = list(base_shape)
        bigger[0] += 2
        t = torch.randn(*bigger, device=DEVICE)
        return t.narrow(0, 1, base_shape[0]).transpose(-2, -1)
    elif pattern == "expanded":
        # Unsqueeze + expand: creates stride-0 dims.
        t = torch.randn(*base_shape, device=DEVICE)
        return t.unsqueeze(0).expand(4, *base_shape)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def run_op_audit(test_case, op_fn, base_shape, pattern, *, n_inputs=1):
    """Run op_fn on patterned inputs and compare to CPU reference."""
    inputs_dev = [make_pattern(base_shape, pattern) for _ in range(n_inputs)]
    inputs_cpu = [t.cpu() for t in inputs_dev]

    result_dev = op_fn(*inputs_dev)
    result_cpu = op_fn(*inputs_cpu)

    test_case.assertEqual(
        result_dev.cpu(), result_cpu,
        msg=f"Mismatch for pattern={pattern}, shape={base_shape}")


class TestUnaryStrideAudit(TestCase):

    OPS = [
        ("abs", torch.abs),
        ("neg", torch.neg),
        ("relu", F.relu),
        ("sigmoid", torch.sigmoid),
        ("tanh", torch.tanh),
        ("exp", torch.exp),
        ("log", lambda t: torch.log(t.abs() + 1)),
        ("sqrt", lambda t: torch.sqrt(t.abs())),
    ]

    def test_unary_contiguous(self):
        for name, op in self.OPS:
            with self.subTest(op=name):
                run_op_audit(self, op, (4, 8), "contiguous")

    def test_unary_transposed(self):
        for name, op in self.OPS:
            with self.subTest(op=name):
                run_op_audit(self, op, (4, 8), "transposed")

    def test_unary_sliced_offset(self):
        for name, op in self.OPS:
            with self.subTest(op=name):
                run_op_audit(self, op, (4, 8), "sliced_offset")

    def test_unary_transposed_offset(self):
        for name, op in self.OPS:
            with self.subTest(op=name):
                run_op_audit(self, op, (4, 8), "transposed_offset")

    def test_unary_expanded(self):
        for name, op in self.OPS:
            with self.subTest(op=name):
                run_op_audit(self, op, (4, 8), "expanded")


class TestBinaryStrideAudit(TestCase):

    OPS = [
        ("add", torch.add),
        ("sub", torch.sub),
        ("mul", torch.mul),
        ("div", torch.div),
        ("maximum", torch.maximum),
    ]

    def _run_binary(self, pattern_lhs, pattern_rhs):
        # Rectangular base shape (4,8). Patterns that transpose swap the
        # visible shape to (8,4). Each operand's base shape must be chosen
        # so the *visible* shapes match after the pattern is applied.
        def base_for(pattern):
            if "transposed" in pattern:
                return (8, 4)
            return (4, 8)

        for name, op in self.OPS:
            with self.subTest(op=name, lhs=pattern_lhs, rhs=pattern_rhs):
                lhs = make_pattern(base_for(pattern_lhs), pattern_lhs)
                rhs = make_pattern(base_for(pattern_rhs), pattern_rhs)
                result_dev = op(lhs, rhs)
                result_cpu = op(lhs.cpu(), rhs.cpu())
                self.assertEqual(result_dev.cpu(), result_cpu)

    def test_both_contiguous(self):
        self._run_binary("contiguous", "contiguous")

    def test_lhs_transposed(self):
        self._run_binary("transposed", "contiguous")

    def test_rhs_transposed(self):
        self._run_binary("contiguous", "transposed")

    def test_both_transposed(self):
        self._run_binary("transposed", "transposed")

    def test_lhs_sliced(self):
        self._run_binary("sliced_offset", "contiguous")

    def test_rhs_sliced(self):
        self._run_binary("contiguous", "sliced_offset")

    def test_lhs_transposed_rhs_sliced(self):
        self._run_binary("transposed", "sliced_offset")

    def test_both_expanded(self):
        self._run_binary("expanded", "expanded")


class TestReductionStrideAudit(TestCase):

    def _run_reduction(self, op_fn, pattern):
        run_op_audit(self, op_fn, (4, 8), pattern)

    def test_sum_transposed(self):
        self._run_reduction(lambda t: t.sum(), "transposed")

    def test_sum_sliced(self):
        self._run_reduction(lambda t: t.sum(), "sliced_offset")

    def test_sum_dim_transposed(self):
        self._run_reduction(lambda t: t.sum(dim=0), "transposed")

    def test_mean_transposed(self):
        self._run_reduction(lambda t: t.mean(), "transposed")

    def test_mean_dim_sliced(self):
        self._run_reduction(lambda t: t.mean(dim=-1), "sliced_offset")


class TestMmStrideAudit(TestCase):

    def test_mm_lhs_transposed(self):
        lhs = torch.randn(8, 4, device=DEVICE).t()  # 4x8
        rhs = torch.randn(8, 6, device=DEVICE)
        result = torch.mm(lhs, rhs)
        self.assertEqual(result.cpu(), torch.mm(lhs.cpu(), rhs.cpu()))

    def test_mm_rhs_transposed(self):
        lhs = torch.randn(4, 8, device=DEVICE)
        rhs = torch.randn(6, 8, device=DEVICE).t()  # 8x6
        result = torch.mm(lhs, rhs)
        self.assertEqual(result.cpu(), torch.mm(lhs.cpu(), rhs.cpu()))

    def test_mm_both_transposed(self):
        lhs = torch.randn(8, 4, device=DEVICE).t()
        rhs = torch.randn(6, 8, device=DEVICE).t()
        result = torch.mm(lhs, rhs)
        self.assertEqual(result.cpu(), torch.mm(lhs.cpu(), rhs.cpu()))

    def test_mm_sliced_offset(self):
        big = torch.randn(6, 8, device=DEVICE)
        lhs = big.narrow(0, 1, 4)  # 4x8 with offset
        rhs = torch.randn(8, 6, device=DEVICE)
        result = torch.mm(lhs, rhs)
        self.assertEqual(result.cpu(), torch.mm(lhs.cpu(), rhs.cpu()))

    def test_bmm_transposed(self):
        lhs = torch.randn(2, 4, 8, device=DEVICE)
        rhs = torch.randn(2, 8, 6, device=DEVICE)
        result = torch.bmm(lhs, rhs)
        self.assertEqual(result.cpu(), torch.bmm(lhs.cpu(), rhs.cpu()))


class TestComparisonStrideAudit(TestCase):

    def test_eq_transposed(self):
        a = make_pattern((4, 8), "transposed")
        b = make_pattern((4, 8), "transposed")
        result = torch.eq(a, b)
        self.assertEqual(result.cpu(), torch.eq(a.cpu(), b.cpu()))

    def test_gt_sliced(self):
        a = make_pattern((4, 8), "sliced_offset")
        b = make_pattern((4, 8), "contiguous")
        result = torch.gt(a, b)
        self.assertEqual(result.cpu(), torch.gt(a.cpu(), b.cpu()))


class TestInplaceStrideAudit(TestCase):

    def test_add_inplace_transposed_self(self):
        x = torch.randn(8, 4, device=DEVICE).t()  # 4x8 transposed
        y = torch.randn(4, 8, device=DEVICE)
        x_cpu = x.cpu().clone()
        y_cpu = y.cpu()
        x.add_(y)
        x_cpu.add_(y_cpu)
        self.assertEqual(x.cpu(), x_cpu)

    def test_mul_inplace_sliced_self(self):
        big = torch.randn(6, 8, device=DEVICE)
        x = big.narrow(0, 1, 4)  # 4x8 with offset
        y = torch.randn(4, 8, device=DEVICE)
        x_cpu = x.cpu().clone()
        y_cpu = y.cpu()
        x.mul_(y)
        x_cpu.mul_(y_cpu)
        self.assertEqual(x.cpu(), x_cpu)


class TestSoftmaxStrideAudit(TestCase):

    def test_softmax_transposed(self):
        x = make_pattern((4, 8), "transposed")
        result = F.softmax(x, dim=-1)
        self.assertEqual(result.cpu(), F.softmax(x.cpu(), dim=-1))

    def test_softmax_sliced(self):
        x = make_pattern((4, 8), "sliced_offset")
        result = F.softmax(x, dim=-1)
        self.assertEqual(result.cpu(), F.softmax(x.cpu(), dim=-1))


if __name__ == "__main__":
    run_tests()
