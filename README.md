# OrderCombo
Source code and data for "Order-aware deep learning for drug combination benefit prediction in cancer cell lines".

![Framework of OrderCombo](https://github.com/liuxuan666/OrderCombo/blob/main/framework.png)  

# Requirements
* Python >= 3.10
* PyTorch >= 2.7
* Transformers >= 4.4
* DeepChem >= 2.4
* RDkit >= 2024.09

Chemberta-zinc base-v1 is a pre-trained deep learning model focused on structural analysis of chemical molecules, which can be downloaded from https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/tree/main.

**It is recommended to use git@github.com:liuxuan666/OrderCombo.git to download the code, as there are two large files under the ChemBERTa-zinc-base-v1 directory: flax_model.msgpack (168 MB) and pytorch_model.bin (171 MB). Please make sure their file sizes are correct.**

The detailed conda environment for configuring the model can be found in "environment.yaml".

# Usage
You can run the following command in Python development environment:

* python main.py \<parameters\>  #---classification task with 5-fold CV

The detailed parameter configuration can be found in the config.yaml file.
