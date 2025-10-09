# Mangrove Monitoring : Machine Learning

This repo includes all development and tools related to the Machine Learning Development of the Mangrove Monitoring Project. Most of our active work is within the drone classification folder. 

We are working on two related projects: Mangrove Area Estimation and Human Activity Segmentation. 

The Mangrove Area Estimation project involves using machine learning models to identify mangroves from drone imagery. The satellite super-resolution project aims to expand the capabilities of our mangrove identification models by enhancing satellite imagery to allow us to precisely track and identify mangroves anywhere on Earth without needing to deploy drones.

The Human Activity project intends to detect human-made structures that pose a threat to mangroves. This includes things like roads and buildings, which each cause harm to the ecosystem. We are currently looking for appropriate datasets that will generalize well to mangrove environs.



### Current Classification Models:

UNet: ResNet18 Encoder (SSL4EO-12), Mangrove Segmentation Decoder

UNet: DenseNet Encoder (Imagenet), Mangrove Segmentation Decoder

SegFormer B0/B2

### Super-Resolution Models being tested:

Schrödinger Bridge Latent Diffusion 

3-layer SRCNN




