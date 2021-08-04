# Imbalanced Image Classification with Complement Cross Entropy (Pytorch)
**[Yechan Kim](https://github.com/unique-chan), [Younkwan Lee](https://github.com/brightyoun), and [Moongu Jeon](https://scholar.google.co.kr/citations?user=zfngGSkAAAAJ&hl=ko&oi=ao)**

[Cite this Paper](https://arxiv.org/abs/2009.02189) (🎉 Our paper is accepted in ***Pattern Recognition Letters*** (IF: 3.756).)

Last modified in **May 5, 2021**.

## This repository contains:
- Training code for image classification
- Proposed loss function for classification 
- Evaluation code for image classification
	- Confusion Matrix (see confusion_matrix.py)
	- t_SNE Embedding (see t_SNE.py)

## Prerequisites
* See REQUIREMENTS.txt

## How to use
1. The directory structure of your dataset should be as follows.
~~~
|—— 📁 your_own_dataset
	|—— 📁 train
		|—— 📁 class_1
			|—— 🖼️ 1.jpg
			|—— ...
		|—— 📁 class_2 
			|—— 🖼️ ...
	|—— 📁 valid
		|—— 📁 class_1
		|—— 📁 ... 
	|—— 📁 test
		|—— 📁 class_1
		|—— 📁 ... 
~~~

2. Run **train.py** for training. The below is an example.
~~~ME
python3 train.py --loss_func='CCE' --gamma=-1 --network_name='efficientnet_b0' --dataset_dir='../svhn' --height=32 --width=32 
--epochs=200 --lr=0.1 --lr_warmup_epochs=5 --mean_std --progress_bar --minus_1_to_plus_1_rescale --gpu_index=1 --store
~~~
See **my_utils/parser.py** for details.


## Contribution
If you find any bugs or have opinions for further improvements, please feel free to create a pull request. All contributions are welcome.

## Reference
1. Hao-Yun Chen, Pei-Hsin Wang, Chun-Hao Liu, Shih-Chieh Chang, Jia-Yu Pan, Yu-Ting Chen, Wei Wei, and Da-Cheng Juan. Complement objective training. arXiv preprint arXiv:1903.01182, 2019.
2. Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Doll ́ar.Focal  loss  for  dense  object  detection. In Proceedings  of  the  IEEE international conference on computer vision, pages 2980–2988, 2017.
3. Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, andMu Li.  Bag of tricks for image classification with convolutional neuralnetworks.  InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 558–567, 2019.
4. https://github.com/calmisential/Basic_CNNs_TensorFlow2
5. https://github.com/Hsuxu/Loss_ToolBox-PyTorch
6. https://github.com/weiaicunzai/pytorch-cifar100
