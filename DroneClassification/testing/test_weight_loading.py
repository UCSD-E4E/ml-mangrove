"""Tests for saving and loading model weights and checkpoints."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from DroneClassification.training_utils.training_utils import TrainingSession


# Setup

class TinySegModel(nn.Module):
    """Fake segmentation model for testing."""

    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        out_ch = num_classes if num_classes > 1 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


def make_session(num_classes=1):
    """Create a TrainingSession with fake data."""
    model = TinySegModel(num_classes=num_classes)
    images = torch.randn(4, 3, 8, 8)
    labels = torch.randint(0, max(num_classes, 2), (4, 8, 8))
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    return TrainingSession(
        model=model,
        trainLoader=loader,
        testLoader=loader,
        lossFunc=nn.MSELoss(),
        num_epochs=1,
        device=torch.device("cpu"),
        save_checkpoints=False,
        metric_mode="none",
    )


def enable_checkpoints(session, save_dir):
    """Turn on checkpointing for a test."""
    session.save_checkpoints = True
    session.experiment_dir = save_dir
    session.training_loss = [0.5]
    session.metrics = [{"Loss": 0.5}]


# A fixed random image
@pytest.fixture
def sample_image():
    torch.manual_seed(42)
    return torch.randn(1, 3, 8, 8)



#  Saving and loading model weights


def test_save_then_load_gives_same_output(sample_image, tmp_path):
    """Save weights, load into a new model — same output?"""
    session_a = make_session()
    session_a.model.eval()
    output_a = session_a.model(sample_image)

    # save A's weights
    session_a.save_model(tmp_path)

    # load them into a completely new model
    session_b = make_session()
    session_b.load_model_weights(tmp_path / "best_model.pth")
    session_b.model.eval()
    output_b = session_b.model(sample_image)

    assert torch.equal(output_a, output_b)


def test_load_fixes_corrupted_weights(sample_image, tmp_path):
    """Trash the weights, load the save — back to normal?"""
    session = make_session()
    session.model.eval()
    original_output = session.model(sample_image).clone()

    session.save_model(tmp_path)

    # trash every weight in the model
    with torch.no_grad():
        for param in session.model.parameters():
            param.zero_()

    trashed_output = session.model(sample_image)
    assert not torch.equal(original_output, trashed_output)

    # load the saved weights back
    session.load_model_weights(tmp_path / "best_model.pth")
    fixed_output = session.model(sample_image)
    assert torch.equal(original_output, fixed_output)


def test_wrong_num_classes_wont_load(tmp_path):
    """1-class weights into a 3-class model — should error."""
    one_class = make_session(num_classes=1)
    one_class.save_model(tmp_path)

    three_class = make_session(num_classes=3)
    with pytest.raises(RuntimeError, match="size mismatch"):
        three_class.load_model_weights(tmp_path / "best_model.pth")


def test_save_actually_creates_the_file(tmp_path):
    session = make_session()
    session.save_model(tmp_path)
    assert (tmp_path / "best_model.pth").exists()


def test_loading_missing_file_errors():
    session = make_session()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        session.load_model_weights("this_file_does_not_exist.pth")



#  Saving and loading checkpoints


def test_checkpoint_saves_epoch_and_metrics(tmp_path):
    """Save at epoch 5, load it — do we get epoch 5 back?"""
    session = make_session()
    enable_checkpoints(session, tmp_path)

    session.save_checkpoint(epoch=5, metrics={"Loss": 0.42, "IoU": 0.73})

    fresh = make_session()
    epoch, metrics = fresh.load_checkpoint(tmp_path / "latest_checkpoint.pth")

    assert epoch == 5
    assert metrics["Loss"] == pytest.approx(0.42)
    assert metrics["IoU"] == pytest.approx(0.73)


def test_checkpoint_restores_model_output(sample_image, tmp_path):
    """Same as the weights test but through a checkpoint."""
    session = make_session()
    enable_checkpoints(session, tmp_path)
    session.model.eval()
    expected = session.model(sample_image)

    session.save_checkpoint(epoch=1, metrics={"Loss": 0.5})

    fresh = make_session()
    fresh.load_checkpoint(tmp_path / "latest_checkpoint.pth")
    fresh.model.eval()
    actual = fresh.model(sample_image)

    assert torch.equal(expected, actual)


def test_checkpoint_restores_optimizer(tmp_path):
    """Optimizer momentum should survive a save/load."""
    session = make_session()
    enable_checkpoints(session, tmp_path)

    # do one training step so the optimizer has some state
    dummy = torch.randn(2, 3, 8, 8)
    loss = session.model(dummy).sum()
    loss.backward()
    session.optimizer.step()

    state_before = session.optimizer.state_dict()
    session.save_checkpoint(epoch=1, metrics={"Loss": 0.5})

    fresh = make_session()
    fresh.load_checkpoint(tmp_path / "latest_checkpoint.pth")
    state_after = fresh.optimizer.state_dict()

    for param_id in state_before["state"]:
        for key, val in state_before["state"][param_id].items():
            restored = state_after["state"][param_id][key]
            if isinstance(val, torch.Tensor):
                assert torch.equal(val, restored)
            else:
                assert val == restored


def test_checkpoint_restores_scheduler(tmp_path):
    """Learning rate schedule should pick up where it left off."""
    session = make_session()
    enable_checkpoints(session, tmp_path)

    for _ in range(3):
        session.scheduler.step()

    state_before = session.scheduler.state_dict()
    session.save_checkpoint(epoch=3, metrics={"Loss": 0.4})

    fresh = make_session()
    fresh.load_checkpoint(tmp_path / "latest_checkpoint.pth")
    state_after = fresh.scheduler.state_dict()

    assert state_before == state_after


def test_best_checkpoint_also_saves_weights(tmp_path):
    """is_best=True should save both the checkpoint and best_model.pth."""
    session = make_session()
    enable_checkpoints(session, tmp_path)

    session.save_checkpoint(epoch=1, metrics={"Loss": 0.5}, is_best=True)

    assert (tmp_path / "latest_checkpoint.pth").exists()
    assert (tmp_path / "best_model.pth").exists()


def test_periodic_checkpoint_every_10_epochs(tmp_path):
    session = make_session()
    enable_checkpoints(session, tmp_path)

    session.save_checkpoint(epoch=10, metrics={"Loss": 0.3})
    assert (tmp_path / "epoch_10_checkpoint.pth").exists()


def test_no_files_when_checkpointing_is_off(tmp_path):
    session = make_session()
    session.save_checkpoints = False
    session.experiment_dir = tmp_path

    session.save_checkpoint(epoch=1, metrics={"Loss": 0.5})
    assert not (tmp_path / "latest_checkpoint.pth").exists()
