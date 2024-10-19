# This is part of the code about Image Classification from my paper **EAN: An Efficient Attention Module Guided by Normalization for Deep Neural Networks**. The paper was accepted to AAAI-2024. The link to the conference is https://ojs.aaai.org/index.php/AAAI/article/view/28093.

# CNNs for image classification
### Requirement
- `pytorch 1.1.0+`
- `torchvision`
- `tensorboard 1.14+`
- `numpy`
- `pyyaml`
- `tqdm`
- `pillow`

### Dataset
- `CIFAR-10`
- `CIFAR-100`
- `ImageNet(2012)`

### Usage
- Add configuration file under `configs` folder
  - `runid` needs to be configured in validate mode and test mode to obtain the model's parameters file
  - If `cuda` is not specified, model use cpu. `cuda` can be specified as `"all"` to use all GPUs, or a list of GPUs, such as `"0,1"`
- run `train.py`, `validate.py` or `test.py`

#Please note that the entire code for the paper is not open-sourced. If you want to use this part of the code, please make sure you refer to the authors and respect our intellectual property. Thank you for your cooperation!
