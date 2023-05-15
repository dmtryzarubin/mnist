import hydra
import torch
from src import LitModel

hydra.initialize(config_path="../config", job_name="test_app")
test_cfg = hydra.compose(config_name="tests")
cfg = hydra.compose(config_name=test_cfg.config_name)


def test():
    """
    Basic testing function
    """
    dm = hydra.utils.instantiate(cfg.datamodule)
    # prepare train/val/test datasets
    dm.setup()
    for name in test_cfg.dataset_names:
        assert hasattr(dm, name)
        dataset = getattr(dm, name)
        assert len(dataset) > 0
        x, y = dataset[0]
        assert isinstance(x, torch.FloatTensor) and isinstance(y, int)
        assert x.ndim == 3

    for name in test_cfg.loaders:
        loader = getattr(dm, name)()
        x, y = next(iter(loader))
        model = LitModel(cfg)
        with torch.no_grad():
            output = model(x)
            preds = output.argmax(1)
        assert preds.shape == y.shape


if __name__ == "__main__":
    test()
