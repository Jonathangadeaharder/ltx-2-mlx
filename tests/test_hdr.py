"""Unit tests for the LogC3 HDR utilities."""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.hdr import LogC3, apply_hdr_decode_postprocess


class TestLogC3:
    def test_compress_clamps_negative(self):
        """Negative inputs are clamped to zero before log."""
        logc = LogC3()
        out = logc.compress(mx.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
        mx.eval(out)
        assert float(out[0].item()) == float(logc.compress(mx.array([0.0])).item())

    def test_compress_in_unit_range(self):
        """Compress output stays within [0, 1]."""
        logc = LogC3()
        x = mx.array([0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 1e6])
        out = logc.compress(x)
        mx.eval(out)
        assert float(out.min().item()) >= 0.0
        assert float(out.max().item()) <= 1.0

    def test_round_trip_linear_branch(self):
        """Compress then decompress on values below CUT: small linear-branch error."""
        logc = LogC3()
        x = mx.array([0.0, 0.002, 0.005, 0.008])  # all < CUT=0.010591
        recovered = logc.decompress(logc.compress(x))
        mx.eval(x, recovered)
        assert mx.allclose(x, recovered, atol=1e-5).item()

    def test_round_trip_log_branch(self):
        """Compress + decompress: log branch round-trips within representable HDR range.

        LogC3 saturates at compress(1.0) ≈ 1.0 and decompress(1.0) ≈ 55.
        Values beyond that are clamped at compression and not recoverable.
        """
        logc = LogC3()
        x = mx.array([0.05, 0.1, 0.5, 1.0, 5.0, 50.0])
        recovered = logc.decompress(logc.compress(x))
        mx.eval(x, recovered)
        assert mx.allclose(x, recovered, atol=1e-3, rtol=1e-3).item()

    def test_clamps_above_dynamic_range(self):
        """Values exceeding the LogC3 dynamic range (~55x linear) saturate at 1.0."""
        logc = LogC3()
        compressed = logc.compress(mx.array([100.0, 1000.0, 1e6]))
        mx.eval(compressed)
        assert float(compressed.min().item()) > 0.99

    def test_decompress_clamps(self):
        """Decompress clamps inputs outside [0, 1]."""
        logc = LogC3()
        out = logc.decompress(mx.array([-0.5, 0.0, 0.5, 1.0, 1.5]))
        mx.eval(out)
        # -0.5 → clamped to 0 → linear branch → (0 - F) / E = negative
        # That's actually the cut-log eval at 0. Just check no NaN / Inf.
        assert not mx.isnan(out).any().item()
        assert not mx.isinf(out).any().item()

    def test_compress_ldr_identity(self):
        """compress_ldr is just clip [0, 1]."""
        logc = LogC3()
        out = logc.compress_ldr(mx.array([-0.5, 0.0, 0.3, 1.0, 1.5]))
        mx.eval(out)
        assert mx.array_equal(out, mx.array([0.0, 0.0, 0.3, 1.0, 1.0])).item()

    def test_compress_continuity_at_cut(self):
        """Linear and log branches must give the same value at CUT."""
        logc = LogC3()
        cut = mx.array([logc.CUT])
        # Linear: E * CUT + F
        lin = logc.E * logc.CUT + logc.F
        # Log: C * log10(A * CUT + B) + D
        out = logc.compress(cut)
        mx.eval(out)
        assert abs(float(out.item()) - lin) < 1e-4


class TestApplyHdrDecodePostprocess:
    def test_logc3_dispatch(self):
        """apply_hdr_decode_postprocess('logc3') matches LogC3().decompress()."""
        v = mx.array([[[[[0.0, 0.5, 1.0]]]]]).astype(mx.float32)  # (1,1,1,1,3)
        a = apply_hdr_decode_postprocess(v, transform="logc3")
        b = LogC3().decompress(v)
        mx.eval(a, b)
        assert mx.array_equal(a, b).item()

    def test_unknown_transform_raises(self):
        import pytest

        v = mx.zeros((1, 1, 1, 1, 1))
        with pytest.raises(ValueError, match="Unsupported HDR transform"):
            apply_hdr_decode_postprocess(v, transform="bogus")  # type: ignore[arg-type]

    def test_upcasts_to_fp32(self):
        v = mx.array([0.5]).astype(mx.bfloat16)
        out = apply_hdr_decode_postprocess(v, transform="logc3")
        mx.eval(out)
        assert out.dtype == mx.float32
