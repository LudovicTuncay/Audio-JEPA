<div align="center">

# Lightning-Hydra-Template

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.6+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

## 📌  Introduction

This repository is based on the very good Lightning-Hydra-Template, which is a template for PyTorch Lightning projects with Hydra configuration.

## 📌  Running the code

This code uses the uv as the package and project manager.
Please refer to the [uv documentation](https://docs.astral.sh/uv/) for more details on installation.

After cloning the repository, navigate to the project directory (`Audio-JEPA`) and run the following command:

```bash
uv sync
```

this will install all the dependencies.

To run the pre-training code, you can use the following command:

```bash
uv run src/train
```

This will run the training script with the default configuration (on CPU).

If you want to change the configuration, you can either change the configurations files directly in the `config` folder or use the command line arguments. For more information, check the [Hydra documentation](https://hydra.cc/docs/intro/).

For example, to run the training script on GPU, logged on WandB, with a specific batch size, and some additional options you could use the following command:

```bash
uv run src/train logger=wandb trainer=gpu trainer.max_steps=100000 data.batch_size=64 callbacks.model_checkpoint.every_n_train_steps=20000 callbacks.model_checkpoint.save_top_k=-1
```

By default, training checkpoints are saved in the `logs/train/runs/<date>` folder. You can change this by modifying the `logger` configuration in the `config` folder. 

To test the model, please refer to our fork of [X-ARES](https://github.com/LudovicTuncay/xares) where we added support for Audio-JEPA.