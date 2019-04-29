# Python-based Deep Learning Frame Server for Nuke

This repository contains the client-server system enabling Deep Learning (DL) inference in Nuke. This work is split into two parts: a client Nuke plug-in [Plugins/Client/](Plugins/Client) and the Python frame server [Plugins/Server](Plugins/Server).

The following models are provided as examples:
- blur: a simple gaussian blur operation
- [Mask-RCNN](https://github.com/facebookresearch/Detectron)
- [DensePose](https://github.com/facebookresearch/DensePose)

<div align="center">
  <img src="https://user-images.githubusercontent.com/27013153/54621337-837f0900-4a5f-11e9-9169-0e8ad1fbe67a.png" width="700px" />
  <p>Example of Nuke doing DensePose inference.</p>
</div>

## Introduction

The Deep Learning (DL) plug-in connects Nuke to a Python server to apply DL models to images.
The plug-in works as follows:
- The Nuke node can connect to a server given an ip address and port,
- The Python server responds with the list of available Deep Learning (DL) models and options,
- The Nuke node displays the models in an enumeration knob, from which the user can choose,
- On every renderStripe call, the current image and model options are sent from the Nuke node to the server,
- The server does an inference on the image using the chosen model/options. This inference can be an actual inference operation of a deep learning model, or just some other image processing code,
- The resulting image is sent back to the Nuke node.

## Installation

Please find installation instructions in [INSTALL.md](INSTALL.md).

## Known issue

The GPU can run out of memory when doing model inference.

To run DensePose or Mask-RCNN, it is necessary to have a GPU memory of at least 6GB.

## License

The source code is licensed under the Apache License, Version 2.0, found in [LICENSE](LICENSE).

Two of the models (Mask RCNN and DensePose) have individual licenses that can be found in their respective folders.

This is strictly non-commercial.

## Contact

- Johanna Barbier (Johanna.Barbier@foundry.com)

This plug-in was initially created by Sebastian Lutz (https://v-sense.scss.tcd.ie/?profile=sebastian-lutz).

## References

- [Mask R-CNN](https://arxiv.org/abs/1703.06870).
  Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick.
  IEEE International Conference on Computer Vision (ICCV), 2017.
- [DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/abs/1802.00434).
  Riza Alp Güler, Natalia Neverova, Iasonas Kokkinos.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.