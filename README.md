## Description

Keras mplementation of adversarial attack methods, using Inception-v3



## Implemented list

1. Fast Gradient Sign Method



## Usage

#### FGSM

In my code, user should prepare NIPS 2017 Competition on Adversarial Attacks and Defenses DEV dataset <https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set>

image.csv, images directory will be in same path(directory).

```python3
python3 generate_data.py
```

And then you will get two files, imageNumpy.npy, labels.npy.

imageNumpy.npy includes data's pixel values and labels.npy include labels.

imageNumpy.npy : (1000, 299, 299, 3) > 1000 images x 299 height x 299 width x 3 channel

labels.npy : (1000, 1) > 1000 labels



I used pretrained Inception-v3 model in experiment.

You can check the original accuracy using following,

```python3
python3 original_acc.py
```

For me, got 0.952



Before check accuracy after FGSM, you should install 'cleverhans' library.

```python3
pip3 install cleverhans
python3 adversary_acc.py
```

I got 0.556 using epsilon 0.2

You can also check actual attacked images after removing '#' in line 50, 51, 52

