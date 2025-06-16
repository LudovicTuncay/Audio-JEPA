from copy import deepcopy
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt


import torch
import wandb
import lightning as L

from src.callbacks.MA_weight_update_callback import MAWeightUpdate
from src.data.components.audioset_dataset import AudioSetBatch
from src.utils.visualize import create_masked_spectrogram
from src.masks.components.utils import apply_masks
from src.optimizers.warmup_cosine import WarmupCosineScheduler
from src.optimizers.cosine_wd import CosineWDScheduler
# from src.optimizers.combined_scheduler import CombinedScheduler

class JEPAModule(L.LightningModule):

    def __init__(
        self,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: WarmupCosineScheduler = None,
        wd_scheduler: CosineWDScheduler = None,
        ma_callback: MAWeightUpdate = MAWeightUpdate(),
        compile: bool = True,
    ) -> None:
        """Initialize a `JEPAModule`.
        
        :param encoder: The model to train.
        :param predictor: The model to predict the target patches.
        :param criterion: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param lr_scheduler: The learning rate scheduler class. Default: None
        :param wd_scheduler: The weight decay scheduler class. Default: None
        :param ma_callback: The callback to update the target encoder at the end of each batch.
        :param compile: Whether to compile the model using `torch.compile`. Default is True.
        """
        
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['encoder', 'predictor', 'criterion'])

        # The encoder model to train. It is the context encoder.
        self.encoder = encoder
        
        # Target encoder as a moving average of the context encoder. It does not require gradients.
        self.target_encoder = deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        # The moving average callback is used to update the target encoder.
        self.ma_callback = ma_callback
           
        # The predictor model. It will predict the target patches
        self.predictor = predictor
        
        # loss function
        self.criterion = criterion

        # optimizer and scheduler
        self.optimizer_cls = optimizer
        
        self.lr_scheduler_cls = lr_scheduler
        self.wd_scheduler_cls = wd_scheduler


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.encoder`.

        :param x: A tensor of spectrograms.
        :return: A tensor representing the embedding of the input spectrograms.
        """
        return self.encoder(x)


    def model_step(
        self, batch: AudioSetBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # print(batch)
        
        spectrograms = batch.spectrograms
        context_masks = batch.context_masks
        prediction_masks = batch.prediction_masks

        # Forward pass through the encoder with context masks
        h = self.encoder(spectrograms, context_masks)

        # Predict the target patches using the predictor
        z_pred = self.predictor(h, context_masks, prediction_masks)

        # Forward pass through the target encoder to get the target embeddings
        with torch.no_grad():
            h_target = self.target_encoder(spectrograms)
            h_target = apply_masks(h_target, prediction_masks)

        # Calculate the loss
        loss = self.criterion(z_pred, h_target)
        
        if loss.isnan():
            def find_nan_indices(tensor):
                """
                Finds the indices of NaN values in a PyTorch tensor.
                
                Args:
                  tensor: The PyTorch tensor to check.
                
                Returns:
                  A list of tuples, where each tuple represents the multi-dimensional
                  index of a NaN value. The length of the tuple corresponds to the
                  number of dimensions of the tensor.
                  Returns an empty list if no NaN values are found.
                """
                nan_mask = torch.isnan(tensor)
                nan_indices = torch.where(nan_mask)
                
                # Transpose the indices to get a list of index tuples
                nan_indices_list = list(zip(*[idx.tolist() for idx in nan_indices]))
                
                return nan_indices_list
            
            # print()
            # print()
            # print("#"*30)
            # print(f"{loss=}")
            
            # nan_indices_spec = find_nan_indices(spectrograms)
            # print(f"{len(nan_indices_spec)=}")
            # print(f"{nan_indices_spec[:3]=}")
            # print(spectrograms[15])
            
            # nan_indices_z_pred = find_nan_indices(z_pred)
            # print(f"{len(nan_indices_z_pred)=}")
            # print(f"{nan_indices_z_pred[:3]=}")
            # print(z_pred[15])
            
            # nan_indices_h_target = find_nan_indices(h_target)
            # print(f"{len(nan_indices_h_target)=}")
            # print(f"{nan_indices_h_target[:3]=}")
            # print(h_target[15])
            # print("#"*30)
            # print()
            # print()
        
        return loss, z_pred, h_target

    def log_specs_with_wandb(self, batch: AudioSetBatch, max_samples: int = 4, batch_idx: int = 0):
        # Log first 4 samples from batch
        
        # Create table structure
        table_columns = [
            "Original Spectrogram", 
            "Context Mask", 
            *[f"Prediction Mask {i+1}/{len(batch.prediction_masks)}" for i in range(len(batch.prediction_masks))],
            "Audio",
        ]

        table_data = []

        for i in range(min(max_samples, batch.spectrograms.size(0))):
            
            spec = batch.spectrograms[i]
            context_mask = batch.context_masks[0][i]
            prediction_masks = [pm[i] for pm in batch.prediction_masks]
            
            # print(i)
            # print(f"{spec.max()=}")
            # print(f"{spec.min()=}")
            # print(f"{spec.mean()=}")
            # print(f"{batch.waveforms[i].mean()=}")
            

    	    # print(f"{batch.prediction_masks}")
            # print(f"{prediction_masks}")
            
            # Create figures
            fig_spectrogram = create_masked_spectrogram(
                spec, 
                patch_t = self.encoder.patch_size[0], patch_mel = self.encoder.patch_size[1], 
                patch_indices=None, 
                show_colorbar=False
            )
            fig_spectrogram.suptitle("Original spectrogram")
            
            fig_context = create_masked_spectrogram(
                spec,
                patch_t = self.encoder.patch_size[0], patch_mel = self.encoder.patch_size[1],
                patch_indices=context_mask,
                show_colorbar=False
            )
            fig_context.suptitle("Context patches")
            
            
            figs_pred = []
            for n, pm in enumerate(prediction_masks):
                fig = create_masked_spectrogram(
                    spec,
                    patch_t = self.encoder.patch_size[0], patch_mel = self.encoder.patch_size[1],
                    patch_indices=pm,
                    show_colorbar=False
                )
            # fig.suptitle(f"Prediction patch {n+1}/{len(prediction_masks)}")
                figs_pred.append(fig)
            
            orig_img = wandb.Image(fig_spectrogram)
            context_img = wandb.Image(fig_context)
            pred_imgs = [wandb.Image(fig) for fig in figs_pred]
            
            # Process audio
            waveform = batch.waveforms[i].cpu().numpy()
            audio = wandb.Audio(waveform, sample_rate=32000)

            # Build table row
            table_row = [orig_img, context_img] + pred_imgs + [audio]
            table_data.append(table_row)

            # Cleanup
            plt.close(fig_spectrogram)
            plt.close(fig_context)
            for fig in figs_pred:
                plt.close(fig)
        # Log table
        self.logger.experiment.log({
            f"spectrogram_samples/batch_{batch_idx}": wandb.Table(
                columns=table_columns,
                data=table_data
            ),
            "global_step": self.global_step
        })
        

    def training_step(
        self, batch: AudioSetBatch, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        
        loss, preds, targets = self.model_step(batch)

        # Log the learning rate with scientific notation
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        current_lr = opt.param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, batch_size=batch.spectrograms.size(0))
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.spectrograms.size(0))
        
        # in the first 2 batches of the first epoch, print the loss, and if the logger is a wandb logger, log
        if isinstance(self.logger, L.pytorch.loggers.wandb.WandbLogger) and self.current_epoch < 2 and batch_idx < 2 and self.trainer.is_global_zero:    
            self.log_specs_with_wandb(batch, max_samples=4, batch_idx=batch_idx)
        return loss

    def validation_step(
        self, batch: AudioSetBatch, batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data containing the spectrograms to be processed.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch.spectrograms.size(0), sync_dist=True)
        self.log("val/variance", torch.var(preds), prog_bar=True, batch_size=batch.spectrograms.size(0), sync_dist=True)
        return

    def test_step(
        self, batch: AudioSetBatch, batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test (eval) set.

        :param batch: A batch of data containing the spectrograms to be processed.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.log("test/loss", loss, prog_bar=True, batch_size=batch.spectrograms.size(0), sync_dist=True)
        self.log("test/variance", torch.var(preds), prog_bar=True, batch_size=batch.spectrograms.size(0), sync_dist=True)
        return 
    
    
    def on_train_batch_end(self, outputs: torch.Tensor, batch: torch.Tensor, batch_idx: int) -> None:
        self.ma_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx)

    # def on_validation_epoch_end(self):
    #     if self.trainer is not None:
    #         metrics = {k: v.item() for k, v in self.trainer.callback_metrics.items()}
    #         print(f"\n📊 Validation Metrics @ step {self.global_step}:\n" + "=" * 30)
    #         for key, value in metrics.items():
    #             print(f"🔹 {key.ljust(15)} | {value:.4f}")
    #         print("=" * 30)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.target_encoder = torch.compile(self.target_encoder)
            self.predictor = torch.compile(self.predictor)


    # @property
    # def num_training_steps(self) -> int:
    #     """Total training steps inferred from datamodule and devices."""
    #     # print(f"max_steps: {self.trainer.max_steps}")
        
    #     # print all methods of self.trainer
    #     print(dir(self.trainer))
        
    #     if self.trainer.max_steps is not None and self.trainer.max_steps > 0:
    #         return self.trainer.max_steps

    #     limit_batches = self.trainer.limit_train_batches
    #     batches = len(self.trainer.train_dataloader)
    #     batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

    #     num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
    #     if self.trainer.tpu_cores:
    #         num_devices = max(num_devices, self.trainer.tpu_cores)

    #     effective_accum = self.trainer.accumulate_grad_batches * num_devices
        
    #     print(f"batches: {batches}, effective_accum: {effective_accum}")
    #     print(f"max_epochs: {self.trainer.max_epochs}")
        
    #     return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self):
        """Configure optimizers and learning-rate schedulers.
        Expects optimizer and scheduler classes to be provided via Hydra config.
        
        Returns:
            dict: Optimizer configuration containing optimizer and scheduler settings.
        """
        # Create parameter groups following the original implementation
        param_groups = [
            {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]
    

        # Initialize optimizer (AdamW as in original implementation)
        optimizer = self.hparams.optimizer(param_groups)

        lr_scheduler = self.hparams.lr_scheduler(
            optimizer=optimizer,
            # T_max=int(self.num_training_steps)
            T_max=self.trainer.estimated_stepping_batches,
        )
        
        wd_scheduler = self.hparams.wd_scheduler(
            optimizer=optimizer,
            # T_max=int(self.num_training_steps)
            T_max=self.trainer.estimated_stepping_batches,
        )
        
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        
        wd_scheduler_config = {
            "scheduler": wd_scheduler,
            "interval": "step",
            "frequency": 1
        }
            
        return [optimizer], [lr_scheduler_config, wd_scheduler_config]
