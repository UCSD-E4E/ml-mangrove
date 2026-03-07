import numpy as np
import torch
import pytest
from DroneClassification.models.utils import class2one_hot, one_hot2dist

class TestClass2OneHot:

    def test_output_shape_2d_input(self):
        """A (H, W) label map should produce a (1, C, H, W) tensor."""
        H, W, C = 4, 4, 3
        seg = torch.randint(0, C, (H, W))
        out = class2one_hot(seg, C)
        assert out.shape == (1, C, H, W), f"Expected (1, {C}, {H}, {W}), got {out.shape}"

    def test_output_shape_batched_input(self):
        """A (B, H, W) label map should produce a (B, C, H, W) tensor."""
        B, H, W, C = 2, 4, 4, 3
        seg = torch.randint(0, C, (B, H, W))
        out = class2one_hot(seg, C)
        assert out.shape == (B, C, H, W), f"Expected ({B}, {C}, {H}, {W}), got {out.shape}"

    def test_output_dtype(self):
        """Output should be int32."""
        seg = torch.randint(0, 3, (4, 4))
        out = class2one_hot(seg, 3)
        assert out.dtype == torch.int32

    def test_exactly_one_hot_channel_per_pixel(self):
        """Each pixel should be active in exactly one channel."""
        C = 4
        seg = torch.randint(0, C, (2, 8, 8))
        out = class2one_hot(seg, C)
        channel_sum = out.sum(dim=1)
        assert (channel_sum == 1).all(), "Each pixel must be hot in exactly one channel"

    def test_correct_channel_is_active(self):
        """The active channel must match the original class index."""
        C = 3
        seg = torch.tensor([[0, 1, 2],
                             [2, 0, 1]])
        out = class2one_hot(seg.unsqueeze(0), C)
        for c in range(C):
            expected = (seg == c).int()
            actual   = out[0, c]
            assert (actual == expected).all(), f"Channel {c} does not match class mask"

    def test_all_same_class(self):
        """If all pixels belong to class k, only channel k should be all-ones."""
        C, k = 5, 3
        seg = torch.full((6, 6), k)
        out = class2one_hot(seg, C)
        for c in range(C):
            if c == k:
                assert out[0, c].sum() == 36
            else:
                assert out[0, c].sum() == 0

    def test_binary_segmentation(self):
        """C=1 edge case: only class 0 exists; channel 0 should be all ones."""
        seg = torch.zeros(4, 4, dtype=torch.long)
        out = class2one_hot(seg, 1)
        assert out.shape == (1, 1, 4, 4)
        assert (out == 1).all()

    def test_values_are_zero_or_one(self):
        """Output should be strictly binary."""
        seg = torch.randint(0, 4, (3, 16, 16))
        out = class2one_hot(seg, 4)
        assert ((out == 0) | (out == 1)).all()


class TestOneHot2Dist:

    @staticmethod
    def _solid_block_mask(H=20, W=20, margin=4) -> np.ndarray:
        """One-hot array (1, H, W) with a filled rectangle in the centre."""
        mask = np.zeros((1, H, W), dtype=np.float32)
        mask[0, margin:H-margin, margin:W-margin] = 1
        return mask

    def test_output_shape_matches_input(self):
        mask = self._solid_block_mask()
        out  = one_hot2dist(mask)
        assert out.shape == mask.shape

    def test_output_dtype_is_float(self):
        mask = self._solid_block_mask()
        out  = one_hot2dist(mask)
        assert np.issubdtype(out.dtype, np.floating)

    def test_exterior_pixels_are_positive(self):
        """Pixels outside the foreground region must have positive distance."""
        mask = self._solid_block_mask()
        out  = one_hot2dist(mask)
        exterior = mask[0] == 0
        assert (out[0][exterior] >= 0).all(), \
            "Exterior pixels should have non-negative distance values"

    def test_interior_pixels_are_non_positive(self):
        """Pixels strictly inside the foreground region must be ≤ 0."""
        mask = self._solid_block_mask(H=30, W=30, margin=8)
        out  = one_hot2dist(mask)
        interior = mask[0] == 1
        assert (out[0][interior] <= 0).all(), \
            "Interior pixels should have non-positive distance values"

    def test_centre_is_farther_than_edge(self):
        """
        The centre pixel of a solid square should be more negative
        (i.e., farther from the boundary) than a pixel on the edge of that square.
        """
        H, W, margin = 40, 40, 4
        mask = self._solid_block_mask(H=H, W=W, margin=margin)
        out  = one_hot2dist(mask)

        edge_val   = out[0, margin, margin]

        centre_row = (margin + H - margin) // 2
        centre_col = (margin + W - margin) // 2
        centre_val = out[0, centre_row, centre_col]

        assert centre_val < edge_val, (
            f"Centre ({centre_val:.2f}) should be more negative than edge ({edge_val:.2f})"
        )

    def test_exterior_distance_increases_away_from_boundary(self):
        """
        Pixels farther from the foreground region should have larger positive values.
        """
        H, W, margin = 40, 40, 10
        mask = self._solid_block_mask(H=H, W=W, margin=margin)
        out  = one_hot2dist(mask)

        just_outside = out[0, margin - 1, margin]
        far_outside  = out[0, 0, 0]

        assert far_outside > just_outside, (
            f"Far exterior ({far_outside:.2f}) should exceed near exterior ({just_outside:.2f})"
        )

    def test_boundary_pixels_near_zero(self):
        """
        The formula is  -(distance(posmask) - 1) * posmask  for interior pixels,
        so the outermost foreground pixels (distance == 1) map to 0.
        """
        mask = self._solid_block_mask()
        out  = one_hot2dist(mask)

        from scipy.ndimage import distance_transform_edt
        dist_fg = distance_transform_edt(mask[0])
        boundary_fg = (mask[0] == 1) & (dist_fg == 1)

        if boundary_fg.any():
            vals = out[0][boundary_fg]
            assert np.allclose(vals, 0, atol=1e-6), \
                f"Boundary foreground pixels should map to 0, got {vals}"

    def test_empty_mask_returns_zeros(self):
        """If a channel has no foreground pixels, distances should remain zero."""
        mask = np.zeros((1, 10, 10), dtype=np.float32)
        out  = one_hot2dist(mask)
        assert (out == 0).all(), "All-background mask should produce all-zero distances"

    def test_multi_class_channels_independent(self):
        """Each channel in a multi-class one-hot should be processed independently."""
        H, W = 20, 20
        mask = np.zeros((2, H, W), dtype=np.float32)
        mask[0, :, :W//2] = 1
        mask[1, :, W//2:] = 1

        out = one_hot2dist(mask)

        assert (out[0, :, W//2:] >= 0).all()
        assert (out[1, :, :W//2] >= 0).all()

class TestPipeline:

    def test_pipeline_produces_valid_distances(self):
        """
        End-to-end: integer label map → one-hot → distance transform.
        Exterior pixels of each class must be non-negative.
        """
        C = 3
        seg = torch.zeros(20, 20, dtype=torch.long)
        seg[5:15, 5:15] = 1
        seg[0:3,  0:3]  = 2

        one_hot = class2one_hot(seg, C)
        one_hot_np = one_hot[0].numpy().astype(np.float32)
        dist = one_hot2dist(one_hot_np)

        for c in range(C):
            exterior = one_hot_np[c] == 0
            assert (dist[c][exterior] >= 0).all(), \
                f"Class {c}: exterior pixels should be non-negative"
