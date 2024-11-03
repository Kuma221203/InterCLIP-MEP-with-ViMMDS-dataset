import yaml
import argparse
import pathlib

from pytorch_lightning import Trainer
from mmsd.lit_model import LitMMSDModel
from mmsd.dataset import MMSDDatasetModule
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def load_yaml_config(yaml_file: str) -> dict:
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def infferent_predict(ckpt_path: str, yaml_config_path: str) -> None:
    config = load_yaml_config(yaml_config_path)
    
    logger = []
    for logger_config in config['trainer']['logger']:
        if logger_config['class_path'] == 'pytorch_lightning.loggers.TensorBoardLogger':
            logger.append(TensorBoardLogger(**logger_config['init_args']))
        elif logger_config['class_path'] == 'pytorch_lightning.loggers.CSVLogger':
            logger.append(CSVLogger(**logger_config['init_args']))

    callbacks = []
    for callback_config in config['trainer']['callbacks']:
        if callback_config['class_path'] == 'pytorch_lightning.callbacks.ModelCheckpoint':
            callbacks.append(ModelCheckpoint(**callback_config['init_args']))

    # Data module initialization from config
    dataloader = MMSDDatasetModule(
        vision_ckpt_name=config['data']['vision_ckpt_name'],
        text_ckpt_name=config['data']['text_ckpt_name'],
        dataset_version=config['data']['dataset_version'],
        train_batch_size=config['data']['train_batch_size'],
        val_batch_size=config['data']['val_batch_size'],
        test_batch_size=config['data']['test_batch_size'],
        num_workers=config['data']['num_workers']
    )

    # Model initialization from config
    model = LitMMSDModel.load_from_checkpoint(ckpt_path)

    # Trainer initialization from config
    trainer = Trainer(
        precision=config['trainer']['precision'],
        logger=logger,
        callbacks=callbacks,
        max_epochs=config['trainer']['max_epochs'],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        enable_checkpointing=config['trainer']['enable_checkpointing'],
        enable_progress_bar=config['trainer']['enable_progress_bar'],
        enable_model_summary=config['trainer']['enable_model_summary'],
        deterministic=config['trainer']['deterministic'],
        use_distributed_sampler=config['trainer']['use_distributed_sampler'],
        default_root_dir=config['trainer']['default_root_dir'],

        # The hyperprameters have been set default value
        
        # fast_dev_run=config['trainer']['fast_dev_run'],
        # overfit_batches=config['trainer']['overfit_batches'],
        # check_val_every_n_epoch=config['trainer']['check_val_every_n_epoch'],
        # accumulate_grad_batches=config['trainer']['accumulate_grad_batches'],
        # detect_anomaly=config['trainer']['detect_anomaly'],
    )

    # Perform prediction using the model and dataloader
    # trainer.predict(model, dataloader)
    print(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", '-ckp', type=pathlib.Path, required=True, help="Checkpoint path for model") 
    parser.add_argument("--config", '-cfg', type=pathlib.Path, required=True, help="Config path has format file .yaml") 
    args = vars(parser.parse_args())
    infferent_predict(
        args['checkpoint'],
        args['config'],
    )
