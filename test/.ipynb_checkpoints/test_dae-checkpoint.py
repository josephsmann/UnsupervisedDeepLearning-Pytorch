import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from udlp.autoencoder.denoisingAutoencoder import DenoisingAutoencoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.002, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../dataset/mnist', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../dataset/mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, num_workers=2)

    in_features = 784
    out_features = 500
    dae = DenoisingAutoencoder(in_features, out_features)
    dae.fit(train_loader, test_loader, lr=args.lr, num_epochs=args.epochs, loss_type="cross-entropy")