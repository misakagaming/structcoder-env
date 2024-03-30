# StructCoder
Official implementation of [StructCoder: Structure-Aware Transformer for Code Generation](https://dl.acm.org/doi/10.1145/3636430)

## Overview
There has been a recent surge of interest in automating software engineering tasks using deep learning. This work addresses the problem of code generation where the goal is to generate target code given source code in a different language or a natural language description.
Most of the state-of-the-art deep learning models for code generation use training strategies primarily designed for natural language. However, understanding and generating code requires a more rigorous comprehension of the code syntax and semantics. With this motivation, we develop an encoder-decoder Transformer model called **StructCoder**, where both the encoder and decoder are explicitly trained to recognize the syntax and data flow in the source and target codes, respectively.
We not only make the encoder structure-aware by leveraging the source code's syntax tree and data flow graph, but we also support the decoder in preserving the syntax and data flow of the target code by introducing two novel auxiliary tasks: AST (Abstract Syntax Tree) paths prediction and data flow prediction. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/42525474/224441090-4fd2195b-d457-4246-a0da-a30a34357a02.png" alt="StructCoder Encoder" width="70%" height="70%">
    
**Structure-aware encoder:** The input sequence to the encoder consists of source code concatenated with the AST leaves and DFG variables, where the AST leaves are embedded using the root-leaf paths in the AST. The modified structure-aware self-attention mechanism of this Transformer encoder utilizes code-AST/DFG linking information, leaf-leaf similarities in the AST, and the (asymmetric) DFG adjacency matrix to compute the attention matrix.
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/42525474/224441103-d1ebb225-55b0-459f-948b-e95fab5be12b.png" alt="StructCoder Encoder" width="50%" height="50%">
    
**Structure-aware decoder:** The decoder generates the next token in the target code as well as predicts the node types on the root-leaf path to the leaf containing this token in the target AST and also the DFG edges incident on this token.
</p>


## Setting up conda environment
    conda create -n structcoder --file structcoder.yml
    conda activate structcoder
For running preprocessing notebooks, add the created structcoder conda enviroment to jupyter notebook using the following commands.

    conda install -c anaconda ipykernel
    python3 -m ipykernel install --user --name=structcoder

## Data Preprocessing
All datasets are loaded from Huggingface's Datasets library except for concode which can be obtained from [here](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code/dataset/concode). Edit the path to Concode dataset in finetune_preprocess.ipynb by replacing '../datasets/concode/'.

Run the cells in pretrain_preprocess.ipynb and finetune_preprocess.ipynb. This should create a folder data/ with subfolders for each dataset used for experiments. You can skip pretrain_preprocess.ipynb if you choose to run our finetuning codes with the provided pretrained checkpoint. 

## Download pretrained checkpoint
Download the pretrained model weights from [here](https://drive.google.com/file/d/10Jee9uv4-XuqecWTlKvo1CeNQh1hOXEs/view?usp=sharing) and place it under saved_models/pretrain/.

## Training commands
#### Pretraining: 

    python pretrain_main.py 
#### Java-C# translation: 

    python finetune_translation_main.py --source_lang java --target_lang cs
#### C#-Java translation: 

    python finetune_translation_main.py --source_lang cs --target_lang java 
#### Concode: 
    
    python finetune_generation_main.py 
#### APPS: 
    
    python finetune_apps_main.py

## Citation
If you find the paper or this repo useful, please cite

Sindhu Tipirneni, Ming Zhu, and Chandan K. Reddy. 2024. StructCoder: Structure-Aware Transformer for Code Generation. ACM Trans. Knowl. Discov. Data 18, 3, Article 70 (April 2024), 20 pages. https://doi.org/10.1145/3636430
