from dataclasses import dataclass

@dataclass
class VitConfig():
    n_gpu: int = 1
    tensorboard: bool = True
    image_size: int = 384
    batch_size: int = 16
    num_workers: int = 2
    train_steps: int = 10000
    lr: float = 1e-3
    wd: float = 1e-4
    warmup_steps: int = 500
    num_classes: int = 10

@dataclass
class VitBase16Config(VitConfig):
    weight_path = './weights/imagenet21k+imagenet2012_ViT-B_16.pth'
    checkpoint_path_cifar10 = './checkpoint/VitBase16_sparse_cifar10.pth'
    checkpoint_path_cifar100 = './checkpoint/VitBase16_sparse_cifar100.pth'
    visualization_path_cifar10 = './visualization/VitBase16/cifar10_sparse'
    visualization_path_cifar100 = './visualization/VitBase16/cifar100_sparse'
    history_path = './outputs/cifar10_sparse_training_history.csv'
    cifar100_history_path = './outputs/cifar100_sparse_training_history.csv'
    patch_size = 16
    emb_dim = 768
    mlp_dim = 3072
    num_heads = 12
    num_layers = 12
    attn_dropout_rate = 0.0
    dropout_rate = 0.1
    
if __name__ == '__main__':
    config = VitBase16Config
    print(config)