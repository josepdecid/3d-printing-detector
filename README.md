# 3D Printing Detector

This repository offers a solution to the challenge presented in
the [HackEPS2021 hackathon](https://lleidahack.dev/hackeps-2021)
by the Invelon - Intech3D company. The task is to build an AI model that is able to identify different pieces created with
3D printers.

## What have we done?

### The Data

We are provided with eight different 3D models (.stl files) representing the eight different objects we have to identify.

### Synthetic Dataset Generation

One of the main problems of this task is the absence of a large dataset with enough data quality and quantity. This idea is
one of the core aspects of most modern AI systems, automatically extracting the features from large
amounts of data. Therefore, we must find a way to generate a Synthetic dataset from the given 3D objects that we can use to train a model that generalizes good enough for real-world environments.

To solve that, we have implemented an online data augmentation pipeline. The *online* term means that the data
augmentation process is done in real-time, every time a sample is loaded in the requested batch. Given the stochasticity of the implement steps, the result is that it potentially creates an infinite number of samples to feed the model with.

### Computer Vision Model

We have decided to apply Deep Convolutional models with [Transfer Learning](https://arxiv.org/abs/1911.02685) techniques
to overcome the data's lack of quantity and quality. We have experimented with
the [ResNet](https://arxiv.org/abs/1512.03385) family model pre-trained with [ImageNet](https://arxiv.org/abs/1409.0575),
experimenting with the number of fine-tuned layers. We also replace the last fully-connected layer with a new linear one with as many neurons as the number of classes we expect.

### Results & Performance Analysis

## Team members:

- [Josep de Cid](https://github.com/josepdecid)
- [Gonzalo Recio](https://github.com/gonzalorecio)
- [Gisela Ruzafa](https://github.com/Gistonic)
- [Pau Torrents](https://github.com/paulovick)
