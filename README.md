# A Novel Diffusion Model for Pairwise Geoscience Data Generation with Unbalanced Training Dataset

## Implementation of UB-Diff

Official Repository for "[A Novel Diffusion Model for Pairwise Geoscience Data Generation with Unbalanced Training Dataset](https://arxiv.org/abs/2501.00941)", accepted by [AAAI 2025](https://aaai.org/conference/aaai/aaai-25/).

<!-- Repository will be updated soon.

UB-Diff addresses the unbalanced issue between different modalities and generates multi-modal pairwise data. We use the geoscience FWI as a case study to generate the pairwise velocity map and seismic wave. -->

## The Framework
![Framework](./imgs/overview.png)


## Dependencies

### Prerequisites
- Python 3.9.9
- PyTorch 1.11.0

### Install the necessary packages.
```bash
pip install -r requirements.txt
```

## Datasets
Dataset can be downloaded at [openFWI](https://smileunc.github.io/projects/openfwi/datasets). 

## Training
### Encoder-Decoder two-steps training
Our code of training encoder and decoder is built upon the [InversionNet](https://github.com/lanl/OpenFWI), https://github.com/lanl/OpenFWI.


### First-step:
Train encoder and decoder for the data from majority group,
e.g., veloctity map in our work. It will use the self-supervised 
learning and large amount of data from majority group to build a good latent.
Run:
```bash
cd encoder_decoder 
python train_EncDec.py 
```
Use the '--dataset' to change the dataset name (choose from "flatvel-a", "curvevel-a", "flatvel-b", "curvevel-b", "style-a" (shown in dataset_config.json)), '--train-data' to set the path of seismic data, 'train-label' to set the path of velocity map,
'--num_data' to set the training velocity map size (since each file of this dataset has 500 smaples, thus the number should be less than the number of files * 500), '--paired_num' to set the paired data size (should less than the number of data), '--fault_fam' to indicate if the dataset is in fault family (default False), '--epoch_block' and '--val_every' to set the epochs and frequency to save checkpoint ('20' used in our experiment).



### Second-step:
In the second step, train the decoder for the data from minority group, 
e.g., seismic waveform in our work. It will train the decoder based on a
good latent representataion. Run:

```bash
python fine_tune_Dec_S.py
```
Use the '--dataset' to change the dataset, '--train-data' to set the path of seismic data, 'train-label' to set the path of velocity map,'--num_data' to set the training velocity map size, '--paired_num' to set the paired data size, '--fault_fam' to indicate if the dataset is in fault family, '--epoch_block' and '--val_every' to set the epochs and frequency to save checkpoint ('10' used in our experiment).


### Diffusion training
Our code for diffusion training is built upon https://github.com/lucidrains/denoising-diffusion-pytorch
and https://github.com/Hramchenko/diffusion_distiller

All the data from the majority group can be used to train the diffusion model.
Run: 
```bash
cd ../diffusion
python train_diff.py
```
Use the '--dataset' to change the dataset, '--num_data' to set the training velocity map size, '--train_data' to set the path of seismic data (not used for training diffusion, just for convenience to load data), 'train_label' to set the path of velocity map.
'num_data' is training velocity map size (24,000, 48,000 and 60,000 are set in our experiment for different datasets).


### Generation
After UB-Diff is trained, run the following to generate velocity map and seismic waveform. The result will be store at 
'gen_data':
```bash
python generation.py
```
Keep a similar setting as train_diff.py.
Use the '--model_file' to change the checkpoint wish to load. '--num_samples' is set default as 500 due to same file size as the original dataset.


## Citation
If you find the code and our idea helpful in your research or work, please cite the following paper.

```
@article{yang2025novel,
  title={A Novel Diffusion Model for Pairwise Geoscience Data Generation with Unbalanced Training Dataset},
  author={Yang, Junhuan and Zhang, Yuzhou and Sheng, Yi and Lin, Youzuo and Yang, Lei},
  journal={arXiv preprint arXiv:2501.00941},
  year={2025}
}
```