import os

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize

from datasets.Piece3DPrintDataset import Piece3DPrintDataset
from transforms.random_background import RandomBackground
from transforms.random_noise import RandomNoise
from transforms.random_offset_scaling import RandomOffsetScalingAndPadding


@hydra.main(config_path=os.path.join('..', 'configs'), config_name='train')
def main(cfg: DictConfig) -> None:
    load_dotenv()

    dataset = Piece3DPrintDataset(os.environ.get('DATASET_PATH'), Compose([
        # Project 3D model into a random-oriented 2D image.
        # RandomProjection(
        #     azimuth=cfg.illumination.azimuth,
        #     altitude=cfg.illumination.altitude,
        #     darkest_shadow_surface=cfg.illumination.darkest_shadow_surface,
        #     brightest_lit_surface=cfg.illumination.brightest_lit_surface,
        # ),

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
        ToTensor(),
        Normalize(mean=cfg.normalization.mean, std=cfg.normalization.std)
    ]))
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device('gpu' if torch.cuda.is_available() and cfg.use_gpu else 'cpu')
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

    for epoch in range(1, 101):
        print(f'Epoch {epoch}')
        correct = 0
        loss_values = []

        for batch_idx, (images, labels) in enumerate(dataloader):
            logits = model(images)

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            predicted_labels = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            correct += (labels == predicted_labels).sum()
            loss_values.append(float(loss.item()))

        print(f'\tAccuracy: {correct}/{len(dataset)} ({correct / len(dataset)})')
        print(f'\tLoss: {sum(loss_values) / len(loss_values)}')


if __name__ == '__main__':
    main()
