# Investigating the Effectiveness of Data Augmentation from Similarity and Diversity: An Empirical Study
This is the official implementation of the [paper](https://www.sciencedirect.com/science/article/pii/S0031320323009019) "Investigating the Effectiveness of Data Augmentation from Similarity and Diversity: An Empirical Study", as was used for the paper.
We release all the codes of our work (both for embedding model training and metric computation).

You can start off using our implementations directly.
## Usage
- Clone this directory and `cd`  into it.
 
`git clone https://github.com/Jackbrocp/Investing_the_Effectiveness_of_Data_Augmentation` 

`cd Investing_the_Effectiveness_of_Data_Augmentation`

## Updates
- 2023/10/03: Initial release

## Getting Started
### Requirements
- Python 3
- PyTorch 1.6.0
- Torchvision 0.7.0
- Numpy
- see requirement.txt
<!-- Install a fitting Pytorch version for your setup with GPU support, as our implementation  -->

### Download the Embedding Models
[ResNet-18](https://drive.google.com/file/d/1fTHi6TiRhaxD3iDgPYcOe7Smt1ZzTaAf/view?usp=drive_link)

[ResNet-50](https://drive.google.com/file/d/1h_87fZF2prm4DHXUkl_6WeGpD7JF0dt4/view?usp=drive_link)

To train the embedding models on your own, please see './Embedding_model_training' folder.
 
### Usage Examples 
This can be seen in the 'Usage_examples.sh'
## Acknowledge 
https://github.com/microsoft/otdd

## Citation
 If you find this repository useful in your research, please cite our paper:
`
 @article{YANG2024110204,
title = {Investigating the effectiveness of data augmentation from similarity and diversity: An empirical study},
journal = {Pattern Recognition},
volume = {148},
pages = {110204},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.110204},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323009019},
author = {Suorong Yang and Suhan Guo and Jian Zhao and Furao Shen}
}
`
