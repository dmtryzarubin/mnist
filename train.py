import warnings

import hydra
import pytorch_lightning as pl
import torch
from src.model import LitModel
from omegaconf import DictConfig, OmegaConf

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

warnings.filterwarnings("ignore")

OmegaConf.register_new_resolver("mul", lambda x, y: int(int(x) * int(y)))


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(cfg: DictConfig) -> None:
    model = LitModel(cfg)
    dm = hydra.utils.instantiate(cfg.datamodule)
    callbacks = [hydra.utils.instantiate(x) for k, x in cfg.callbacks.items()]
    trainer = hydra.utils.instantiate(cfg.trainer)(callbacks=callbacks)
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
