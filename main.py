import torch
import torch.nn as nn

import config
import data_loaders
import utils
import model
from train import train
from test import test
from prune import prune_ViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = config.VitBase16Config
vitbase16_model = model.VisionTransformer(
            image_size=(config.image_size, config.image_size),
            patch_size=(config.patch_size, config.patch_size),
            emb_dim=config.emb_dim,
            mlp_dim=config.mlp_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            attn_dropout_rate=config.attn_dropout_rate,
            dropout_rate=config.dropout_rate)

train_loader = data_loaders.get_train_loader(image_size=config.image_size)
test_loader = data_loaders.get_test_loader(image_size = config.image_size)

state_dict = torch.load(config.weight_path)['state_dict']
del state_dict['classifier.weight']
del state_dict['classifier.bias']
vitbase16_model.load_state_dict(state_dict, strict=False)
vitbase16_model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    params=vitbase16_model.parameters(),
    lr=config.lr,
    weight_decay=config.wd,
    momentum=0.9)

epochs = 1

config.train_steps = len(train_loader) * epochs
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=config.lr,
    pct_start=config.warmup_steps / config.train_steps,
    total_steps=config.train_steps)

prune = True

if __name__ == '__main__':
    train(vitbase16_model,
          config.checkpoint_path_cifar10,
          train_loader,
          criterion, optimizer,
          lr_scheduler, epochs,
          device, config.history_path)
    
    best_weight = torch.load(config.checkpoint_path_cifar10)
    vitbase16_model.load_state_dict(best_weight['model_state_dict'], strict=False)
    
    test(vitbase16_model, test_loader, device)
    
    if prune:
        pruned_model = prune_ViT(vitbase16_model, 0.1)
        
    print('Testing after prune\n')
    test(vitbase16_model, test_loader, device)