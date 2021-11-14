import sys
import argparse
import yaml
import pathlib

import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

from monai.networks.nets import UNETR
from monai.data import DataLoader

sys.path.append('../src/')
import dataset
import transforms
import losses
import metrics
import trainer
import utils


def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # read config:
    path_to_data = pathlib.Path(config['path_to_data'])
    path_to_pkl = pathlib.Path(config['path_to_pkl'])
    path_to_save_dir = pathlib.Path(config['path_to_save_dir'])

    train_batch_size = int(config['train_batch_size'])
    val_batch_size = int(config['val_batch_size'])
    num_workers = int(config['num_workers'])
    lr = float(config['lr'])
    n_epochs = int(config['n_epochs'])

    out_channels = int(config['out_channels'])
    in_channels = int(config['in_channels'])
    
    T_0 = int(config['T_0'])
    eta_min = float(config['eta_min'])

    # train and val data paths:
    all_paths = utils.get_paths_to_patient_files(path_to_imgs=path_to_data, append_mask=True)
    train_paths, val_paths = utils.get_train_val_paths(all_paths=all_paths, path_to_train_val_pkl=path_to_pkl)

    # train and val data transforms:
    train_transforms = transforms.Compose([
        transforms.RandomRotation(p=0.5, angle_range=[0, 45]),
        transforms.Mirroring(p=0.5),
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])

    # datasets:
    train_set = dataset.HecktorDataset(train_paths, transforms=train_transforms)
    val_set = dataset.HecktorDataset(val_paths, transforms=val_transforms)

    # dataloaders:
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    model = UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=(144, 144, 144),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )

    criterion = losses.Dice_and_FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    metric = metrics.dice
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)

    trainer_ = trainer.ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        metric=metric,
        scheduler=scheduler,
        num_epochs=n_epochs,
        parallel=True
    )

    trainer_.train_model()
    trainer_.save_results(path_to_dir=path_to_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)
