from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
torch.set_float32_matmul_precision('high')
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a test set, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multi-runs, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    spectro_config = cfg.data.transforms[0]    
    n_mels = spectro_config.n_mels
    time_bins = spectro_config.target_time_bins
    # Calculate the dimensions of the patched spectrogram
    patch_h = cfg.model.encoder.patch_size[0]
    patch_w = cfg.model.encoder.patch_size[1]
    H = time_bins // patch_h
    W = n_mels // patch_w
    log.info(f"{time_bins=}, {n_mels=}")
    log.info(f"{patch_h=}, {patch_w=}")
    log.info(f"{H=}, {W=}")

    log.info(f"Instantiating mask collator <{cfg.masks._target_}>")
    mask_collator = hydra.utils.instantiate(cfg.masks, input_size=(time_bins, n_mels), patch_size=(patch_h, patch_w))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, mask_collator=mask_collator, seed=cfg.get("seed"))
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Update encoder config with the input size determined by the datamodule
    encoder_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.encoder, resolve=True))
    encoder_cfg.input_size = (time_bins, n_mels)
    
    predictor_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.predictor, resolve=True))
    predictor_cfg.num_patches_h = H
    predictor_cfg.num_patches_w = W
    
    # First initialize the encoder
    log.info(f"Instantiating encoder <{cfg.model.encoder._target_}>")
    encoder = hydra.utils.instantiate(encoder_cfg)
    
    # Initialize predictor with updated config
    log.info(f"Instantiating predictor <{predictor_cfg._target_}>")
    predictor = hydra.utils.instantiate(predictor_cfg)
    
    # Initialize criterion and other components
    log.info(f"Instantiating criterion <{cfg.model.criterion._target_}>")
    criterion = hydra.utils.instantiate(cfg.model.criterion)
    
    # Create model config without components that are already instantiated
    model_cfg = OmegaConf.create({
        "_target_": cfg.model._target_,
        "compile": cfg.model.get("compile", False),
        "optimizer": cfg.model.optimizer,
        "lr_scheduler": cfg.model.lr_scheduler,
    })
    
    log.info(f"Instantiating model <{model_cfg._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        model_cfg,
        encoder=encoder,
        predictor=predictor,
        criterion=criterion
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))


    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    
    fast_dev_run = cfg.get("fast_dev_run", False)
    
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, fast_dev_run=fast_dev_run)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)


    # datamodule.setup()
    # log.info("Testing components!")
    # # get one batch of data
    # batch = next(iter(datamodule.train_dataloader()))
    # # test the model
    # log.info(batch)
    
    # # display the spectrogram
    # import matplotlib.pyplot as plt
    # # subplots
    # batch_size = cfg.data.batch_size
    # # find the closest square number
    # n_rows = int(batch_size ** 0.5)
    # n_cols = n_rows
    
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4))
    
    # for i in range(batch_size):
    #     row = i // n_cols
    #     col = i % n_cols
    #     spectrogram = batch.spectrograms[i][0]
    #     print(f"{spectrogram.shape=}")
    #     axs[row, col].imshow(batch.spectrograms[i][0].T, aspect='equal', origin='lower')
    # plt.show()
    
    # print(model.model_step(batch))
    
    # exit()
    
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        log.info("Final validation epoch")
        trainer.validate(model=model, datamodule=datamodule)
        
    train_metrics = trainer.callback_metrics
    
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
