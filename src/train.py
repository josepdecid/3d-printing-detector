import os

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from datasets.Piece3DPrintDataset import Piece3DPrintDataset
from transforms.random_background import RandomBackground
from transforms.random_noise import RandomNoise
from transforms.random_offset_scaling import RandomOffsetScalingAndPadding
from transforms.random_projection import RandomProjection
from utils import set_seeds


@hydra.main(config_path=os.path.join('..', 'configs'), config_name='train')
def main(cfg: DictConfig) -> None:
    load_dotenv()
    set_seeds(cfg.seed)

    training_dataset = Piece3DPrintDataset(os.environ.get('DATASET_PATH'), Compose([
        # Project 3D model into a random-oriented 2D image.
        RandomProjection(),

        # Apply random transformations to the image to scale, crop and offset.
        RandomOffsetScalingAndPadding(target_size=(
            cfg.img_size.w,
            cfg.img_size.h
        )),

        # Add a randomized background with different cropping sizes.
        RandomBackground(
            search_terms=cfg.backgrounds.search_terms,
            download_bg=cfg.backgrounds.download,
            images_per_term=cfg.backgrounds.images_per_term
        ),

        # Add perturbations with occlusions and random noise to provide robustness to the model.
        RandomNoise(),

        # Convert the image as a Tensor Normalize to [-1, 1] range to input it to the NN model.
        Resize(size=(cfg.img_size.w, cfg.img_size.h)),
        ToTensor(),
        Normalize(mean=cfg.normalization.mean, std=cfg.normalization.std)
    ]))

    eval_dataset = Piece3DPrintDataset(os.environ.get('EVAL_DATASET_PATH'), Compose([
        ToTensor(),
        Normalize(mean=cfg.normalization.mean, std=cfg.normalization.std)
    ]), eval_mode=True)

    training_loader = DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_gpu else 'cpu')
    model = models.resnet18(pretrained=True)
    print(f'Using pre-trained {model.__class__.__name__} from ImageNet.')

    # Apply transfer learning
    ct = 0
    for child in model.children():
        ct += 1
        if ct <= cfg.layers_to_freeze:
            for param in child.parameters():
                param.requires_grad = False
        else:
            print(f'\t > Layers from {ct} onwards will be fine-tuned.')
            break

    # Replace last layer with a #classes linear layer
    model.fc = nn.Linear(512, cfg.num_classes)
    print(f'\t > Substituting the FC layer with a new one with {cfg.num_classes} neurons.\n')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_checkpoint_loss = torch.inf

    for epoch in range(1, 101):
        print(f'Epoch {epoch}')
        correct = 0
        loss_values = []

        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            predicted_labels = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            correct += (labels == predicted_labels).sum()
            loss_values.append(float(loss.item()))

        mean_loss = sum(loss_values) / len(loss_values)
        print(f'\t[T] Loss: {mean_loss}')
        print(f'\t[T] Accuracy: {correct}/{len(training_dataset)} ({correct / len(training_dataset)})')

        if epoch % cfg.epochs_to_eval == 0:
            model.eval()

            correct_test = 0
            loss_test_values = []

            with torch.no_grad():
                for images, labels in eval_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    loss = criterion(logits, labels)

                    predicted_labels = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    correct_test += (labels == predicted_labels).sum()
                    loss_test_values.append(float(loss.item()))

                mean_test_loss = sum(loss_test_values) / len(loss_test_values)
                print(f'\t[E] Loss: {mean_test_loss}')
                print(f'\t[E] Accuracy: {correct_test}/{len(eval_dataset)} ({correct_test / len(eval_dataset)})')

            if mean_test_loss < best_checkpoint_loss:
                best_checkpoint_loss = mean_test_loss
                torch.save(model.state_dict(), 'best_checkpoint.pt')

            model.train()


if __name__ == '__main__':
    main()
