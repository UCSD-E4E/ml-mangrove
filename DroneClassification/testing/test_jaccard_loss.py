import torch
import pytest

from DroneClassification.models.loss import JaccardLoss

# -------------------------
# Binary segmentation tests
# -------------------------
def test_jaccard_loss_binary_perfect_prediction():
    """
    Tests the case where the prediction perfectly matches the target.

    Because the binary case uses a sigmoid, the probabilities never reach
    exactly 0 or 1, so the Jaccard loss will not be exactly 0. Instead,
    we assert that the loss is very close to zero.
    """
    loss_fn = JaccardLoss(num_classes=1, alpha=0.5)

    logits = torch.tensor(
        [[[[10.0, -10.0],
           [-10.0, 10.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor(
        [[[1, 0],
          [0, 1]]],
        dtype=torch.long,
    )

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() < 1e-4


def test_jaccard_loss_binary_wrong():
    """
    Tests the case where the prediction perfectly misses the target.

    We don't check if the loss is close to 1 because the final loss
    is a mixture of BCE and Jaccard loss. The BCE component can be larger
    than 1 depending on the logits and alpha weighting, so the overall loss
    is not guaranteed to be near 1. We instead verify correct behavior by
    ensuring it has a greater loss than a perfect prediction.  
    """
    loss_fn = JaccardLoss(num_classes=1, alpha=0.5)

    labels = torch.tensor(
        [[[1, 0],
          [0, 1]]],
        dtype=torch.long,
    )

    logits_perfect = torch.tensor(
        [[[[10.0, -10.0],
           [-10.0, 10.0]]]],
        dtype=torch.float32,
    )
    logits_wrong = torch.tensor(
        [[[[-10.0, 10.0],
           [10.0, -10.0]]]],
        dtype=torch.float32,
    )

    perfect_loss = loss_fn(logits_perfect, labels)
    wrong_loss = loss_fn(logits_wrong, labels)

    assert wrong_loss.item() > perfect_loss.item()


def test_jaccard_loss_binary_ignore_index_is_ignored():
    """
    Verify that pixels labeled with ignore_index are excluded from loss computation.

    Two predictions are constructed that differ only with one pixel (ignored). 
    Since ignored pixels should not contribute to either the BCE or IoU 
    components, both predictions should produce the same loss value.
    """
    loss_fn = JaccardLoss(num_classes=1, ignore_index=255, alpha=0.5)

    labels = torch.tensor(
        [[[1, 255],
          [0, 1]]],
        dtype=torch.long,
    )

    logits_a = torch.tensor(
        [[[[10.0, -10.0],
           [-10.0, 10.0]]]],
        dtype=torch.float32,
    )
    logits_b = torch.tensor(
        [[[[10.0, 10.0],   # only ignored pixel differs
           [-10.0, 10.0]]]],
        dtype=torch.float32,
    )

    loss_a = loss_fn(logits_a, labels)
    loss_b = loss_fn(logits_b, labels)

    assert torch.allclose(loss_a, loss_b, atol=1e-6)

# ----------------------------
# Multi-class segmentation tests
# ----------------------------
def test_jaccard_loss_multiclass_perfect_prediction():
    """
    Tests the case where the prediction perfectly matches the target.

    Because the multi-class uses a softmax, the probabilities never reach
    exactly 0 or 1 for finite logits. The Jaccard loss will not be 
    exactly 0. Instead, we assert that the loss is very close to zero.
    """
    loss_fn = JaccardLoss(num_classes=3, alpha=0.5)

    logits = torch.tensor([[
        [[10.0, -10.0],
         [-10.0, -10.0]],

        [[-10.0, 10.0],
         [-10.0, 10.0]],

        [[-10.0, -10.0],
         [10.0, -10.0]],
    ]], dtype=torch.float32, requires_grad=True)

    labels = torch.tensor(
        [[[0, 1],
          [2, 1]]],
        dtype=torch.long,
    )

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() < 1e-4


def test_jaccard_loss_multiclass_wrong():
    """
    Tests the case where the prediction perfectly misses the target.

    We don't check if the loss is close to 1 because the final loss
    is a weighted combination of Cross-Entropy and Jaccard loss. The
    Cross-Entropy component can exceed 1, so the combined loss is not
    guaranteed to be near 1. 
    """
    loss_fn = JaccardLoss(num_classes=3, alpha=0.5)

    labels = torch.tensor(
        [[[0, 1],
          [2, 1]]],
        dtype=torch.long,
    )

    logits_perfect = torch.tensor([[
        [[10.0, -10.0],
         [-10.0, -10.0]],

        [[-10.0, 10.0],
         [-10.0, 10.0]],

        [[-10.0, -10.0],
         [10.0, -10.0]],
    ]], dtype=torch.float32)

    logits_wrong = torch.tensor([[
        [[-10.0, 10.0],
         [10.0, 10.0]],

        [[10.0, -10.0],
         [10.0, -10.0]],

        [[10.0, 10.0],
         [-10.0, 10.0]],
    ]], dtype=torch.float32)

    perfect_loss = loss_fn(logits_perfect, labels)
    wrong_loss = loss_fn(logits_wrong, labels)

    assert wrong_loss.item() > perfect_loss.item()


def test_jaccard_loss_multiclass_ignore_index_is_ignored():
    """
    Verify that ignored_index pixels excluded from loss computation 
    for multi-class segmentation.
    """
    loss_fn = JaccardLoss(num_classes=3, ignore_index=255, alpha=0.5)

    labels = torch.tensor(
        [[[0, 255],
          [2,   1]]],
        dtype=torch.long,
    )

    logits_a = torch.tensor([[
        [[10.0, -10.0],
         [-10.0, -10.0]],

        [[-10.0, -10.0],
         [-10.0, 10.0]],

        [[-10.0, -10.0],
         [10.0, -10.0]],
    ]], dtype=torch.float32)

    logits_b = torch.tensor([[
        [[10.0, 10.0],      # only ignored pixel changed
         [-10.0, -10.0]],

        [[-10.0, -10.0],
         [-10.0, 10.0]],

        [[-10.0, -10.0],
         [10.0, -10.0]],
    ]], dtype=torch.float32)

    loss_a = loss_fn(logits_a, labels)
    loss_b = loss_fn(logits_b, labels)

    assert torch.allclose(loss_a, loss_b, atol=1e-6)

# ----------------------------
# backward() tests
# ----------------------------
def test_jaccard_loss_binary_backward():
    """
    Ensure binary mode JaccardLoss supports backpropagation.
    """
    loss_fn = JaccardLoss(num_classes=1, alpha=0.5)

    logits = torch.randn(1, 1, 2, 2, requires_grad=True)
    labels = torch.tensor(
        [[[1, 0],
          [0, 1]]],
        dtype=torch.long,
    )

    loss = loss_fn(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_jaccard_loss_multiclass_backward():
    """
    Ensure multiclass JaccardLoss supports backpropagation.
    """
    loss_fn = JaccardLoss(num_classes=3, alpha=0.5)

    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    labels = torch.tensor(
        [[[0, 1],
          [2, 1]]],
        dtype=torch.long,
    )

    loss = loss_fn(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

# ----------------------------
# Alpha tests
# ----------------------------
def test_jaccard_loss_alpha_changes_output():
    """
    Tests that passing in different alpha values affects the loss 
    computation regarding the weighting of the BCE/CE and pure
    Jaccard loss components.
    """
    labels = torch.tensor(
        [[[1, 0],
          [0, 1]]],
        dtype=torch.long,
    )

    logits = torch.tensor(
        [[[[2.0, -1.0],
           [0.5, -0.5]]]],
        dtype=torch.float32,
    )

    loss_alpha_0 = JaccardLoss(num_classes=1, alpha=0.0)(logits, labels)
    loss_alpha_1 = JaccardLoss(num_classes=1, alpha=1.0)(logits, labels)

    assert torch.isfinite(loss_alpha_0)
    assert torch.isfinite(loss_alpha_1)
    assert not torch.allclose(loss_alpha_0, loss_alpha_1)

# ----------------------------
# Label shape compatibility tests
# ----------------------------
def test_jaccard_loss_binary_label_shape():
    """
    Ensures that binary mode JaccardLoss accepts label shapes of 
    [B,1,H,W] and [B,H,W]
    """
    loss_fn = JaccardLoss(num_classes=1, alpha=0.5)

    logits = torch.tensor(
        [[[[2.0, -2.0],
           [-2.0, 2.0]]]],
        dtype=torch.float32,
    )

    labels_bhw = torch.tensor(
        [[[1, 0],
          [0, 1]]],
        dtype=torch.long,
    )

    labels_b1hw = labels_bhw.unsqueeze(1)  # [B, 1, H, W]

    loss_bhw = loss_fn(logits, labels_bhw)
    loss_b1hw = loss_fn(logits, labels_b1hw)

    assert torch.isfinite(loss_bhw)
    assert torch.isfinite(loss_b1hw)
    assert torch.allclose(loss_bhw, loss_b1hw, atol=1e-6)


def test_jaccard_loss_multiclass_label_shape():
    """
    Ensures that multiclass mode JaccardLoss accepts label shapes of 
    [B,1,H,W] and [B,H,W]
    """
    loss_fn = JaccardLoss(num_classes=3, alpha=0.5)

    logits = torch.tensor([[
        [[10.0, -10.0],
         [-10.0, -10.0]],

        [[-10.0, 10.0],
         [-10.0, 10.0]],

        [[-10.0, -10.0],
         [10.0, -10.0]],
    ]], dtype=torch.float32)

    labels_bhw = torch.tensor(
        [[[0, 1],
          [2, 1]]],
        dtype=torch.long,
    )

    labels_b1hw = labels_bhw.unsqueeze(1)  # [B, 1, H, W]

    loss_bhw = loss_fn(logits, labels_bhw)
    loss_b1hw = loss_fn(logits, labels_b1hw)

    assert torch.isfinite(loss_bhw)
    assert torch.isfinite(loss_b1hw)
    assert torch.allclose(loss_bhw, loss_b1hw, atol=1e-6)