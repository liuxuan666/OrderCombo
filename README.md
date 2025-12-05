# OrderCombo
Source code and data for "Order-aware deep learning for drug combination benefit prediction in cancer cell lines"

![Framework of OrderCombo](https://github.com/liuxuan666/OrderCombo/blob/main/framework.png)  

# Requirements
* Python >= 3.10
* PyTorch >= 1.7
* Transformer >= 3.4
* DeepChem >= 2.4
* RDkit >= 2020.09

Chemberta-zinc base-v1 is a pre-trained deep learning model focused on structural analysis of chemical molecules, which can be downloaded from https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/tree/main.

The detailed conda environment for configuring the model can be found in environment.yaml

# Usage
You can run the following command in python development environment:

* python main.py \<parameters\>  #---classification task with 5-fold CV
