import hydra
import torch
from src import LitModel

hydra.initialize(config_path="../config", job_name="test_app")
test_cfg = hydra.compose(config_name="tests")
cfg = hydra.compose(config_name=test_cfg.config_name)


def test_model():
    model = LitModel(cfg).eval()
    assert hasattr(model, "criterion")
    shape = (16, 1, 32, 32)
    inp = torch.rand(*shape)
    target = torch.randint(cfg.num_classes, shape[:1])
    with torch.no_grad():
        logits = model(inp)
    assert logits.shape == (shape[0], cfg.num_classes)
    loss = model.criterion(logits, target)


if __name__ == "__main__":
    test_model()
