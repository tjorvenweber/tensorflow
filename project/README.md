# Face-Aging with Conditional GANs

This project aimed at rebuilding the face-aging and -rejuvenation technique proposed in "Face Aging with conditional GANs" by Antipov et al. 
(https://ieeexplore.ieee.org/abstract/document/8296650?casa_token=9b5qUiOyCjgAAAAA:PIwhX1uVbNoWMKjovCJtfACLxalMDHleBKOKY3phQl5PWpjHHSglVxC4KITGXu6rDViY2EM0zdM)


## Requirements

```
tensorflow-probability 
numpy 
pandas
matplotlib
yaml
scipy
```

## Data

The project uses the IMDB Dataset, containing 460 723 faces from 20 celebrities. The dataset with the corresponding can be downloaded from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/.

## Usage 

### Model Training

The training for both the ageCGAN and the encoder and the optimization of the latent vector can be run with the corresponding bash files or, alternatively, with the following command

```
python train.py --model <model> --config <config-file>
```

### Face Aging & Rejuvenation
he face-aging or -rejuvenation can be applied with 

```
python change_age.py --image <image-paht> --age <desired age>
```
