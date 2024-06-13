# Semi-supervised Learning for Labeling Veterinary X-Ray Data

This repository contains code for the paper
**"Semi-supervised Learning for Labeling Veterinary X-Ray Data"** 
by Joongchan Doyun Kwon, Ahn, Joon Im, Jeongmin Son, Eunbi Na

## Dependencies

* `python3`
* `pytorch == 1.1.0`
* `torchvision`
* `progress`
* `scipy`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`

## Scripts
Please check out `run_vet.sh` for the scripts to run the baseline algorithms and ours (DARP).

### Training procedure
Train a network with baseline algorithm, e.g., MixMatch on Veterinary data
```
python3.8 vet.py --semi_method mix --gpu 2 --dataset vet --ratio 4 --num_max 500
--imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8

```

Applying DARP on the baseline algorithm
```
python3.8 vet.py --semi_method mix --gpu 2 --dataset vet --darp --align --alpha 2 --warm 250 --ratio 4 --num_max 500
--imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8

```

Train a network with baseline algorithm, e.g., FixMatch on Veterinary data
```
python3.8 vet.py --semi_method fix --gpu 2 --dataset vet --align --ratio 4 --num_max 500
--imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8

```

Applying Laplace-gaussian filter and Sobel filter for strong augmentation on X-ray data
```
python3.8 vet_ours.py --change_aug yes --semi_method fix --gpu 2 --dataset vet --align --ratio 4 --num_max 500
--imb_ratio_l 100 --imb_ratio_u 1 --epochs 150 --val-iteration 771 --batch-size 8

```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project includes code from the following project(s):

- [DARP](https://github.com/bbuing9/DARP): Copyright (c) 2020 bbuing9, Licensed under the MIT License.


