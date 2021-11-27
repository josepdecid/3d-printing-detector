import copy
import os

import hydra
import torch
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from datasets.Piece3DPrintDataset import Piece3DPrintDataset
from src.utils import get_cnn_model
from transforms.random_background import RandomBackground
from transforms.random_projection import RandomProjection


@hydra.main(config_path=os.path.join('..', 'configs'), config_name='train')
def main(cfg: DictConfig) -> None:
    load_dotenv()

    dataset = Piece3DPrintDataset(os.environ.get('DATASET_PATH'), Compose([
        # Project 3D model into a random-oriented 2D image
        RandomProjection(
            azimuth=cfg.illumination.azimuth,
            altitude=cfg.illumination.altitude,
            darkest_shadow_surface=cfg.illumination.darkest_shadow_surface,
            brightest_lit_surface=cfg.illumination.brightest_lit_surface,
        ),

        # Add a randomized background with different cropping sizes
        RandomBackground(
            search_terms=cfg.backgrounds.search_terms,
            download_bg=cfg.backgrounds.download,
            images_per_term=cfg.backgrounds.images_per_term
        ),

        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]))

    dataloader = DataLoader(dataset, 1)
    cnn_model, input_size = get_cnn_model()
    phase = 'train'
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    is_inception = False

    params_to_update = cnn_model.parameters()
    for name, param in cnn_model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    running_loss = 0.0
    running_corrects = 0
    val_acc_history = []

    best_model_wts = copy.deepcopy(cnn_model.state_dict())
    best_acc = 0.0
    num_epochs = 3
    for epoch in range(3):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_loss = 0
        epoch_acc = 0
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn_model.train()  # Set model to training mode
            else:
                cnn_model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for img, label in dataloader:
                plt.imshow(img[0, :, :, :].permute(1, 2, 0).numpy())
                # cnn_model(img.to(torch.float))
                plt.show()

                inputs = img.to(torch.float).to(device)
                labels = label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = cnn_model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = cnn_model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(cnn_model.state_dict())
        if phase == 'val':
            val_acc_history.append(epoch_acc)


    print()

    #time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    cnn_model.load_state_dict(best_model_wts)
    return cnn_model, val_acc_history

if __name__ == '__main__':
    main()
