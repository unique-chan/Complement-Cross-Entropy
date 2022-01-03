# Imbalanced Image Classification with Complement Cross Entropy (Pytorch)
**[Yechan Kim](https://github.com/unique-chan), [Younkwan Lee](https://github.com/brightyoun), and [Moongu Jeon](https://scholar.google.co.kr/citations?user=zfngGSkAAAAJ&hl=ko&oi=ao)**

[Cite this Paper](https://doi.org/10.1016/j.patrec.2021.07.017) (🎉 Our paper is accepted to ***Pattern Recognition Letters***.)

## This repository contains:
- Complement Cross Entropy (code) 
- For simplicity, classification code is provided separately in this [GitHub repo 🖱️](https://github.com/unique-chan/Simple-Image-Classification): you can easily use `Complement Cross Entropy` by passing `--loss_function='CCE'` for executing `train.py`. For details, please visit the above repository.

## Prerequisites
* See REQUIREMENTS.txt
```
torch
torchvision
```

## Code
```python
class CCE(nn.Module):
    def __init__(self, device, balancing_factor=1):
        super(CCE, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.device = device # {'cpu', 'cuda:0', 'cuda:1', ...}
        self.balancing_factor = balancing_factor

    def forward(self, yHat, y):
        # Note: yHat.shape[1] <=> number of classes
        batch_size = len(y)
        # cross entropy
        cross_entropy = self.nll_loss(F.log_softmax(yHat, dim=1), y)
        # complement entropy
        yHat = F.softmax(yHat, dim=1)
        Yg = yHat.gather(dim=1, index=torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(batch_size, yHat.shape[1]).scatter_(
            1, y.view(batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.to(device=self.device)
        complement_entropy = torch.sum(output) / (float(batch_size) * float(yHat.shape[1]))

        return cross_entropy - self.balancing_factor * complement_entropy
```


## Contribution
If you find any bugs or have opinions for further improvements, please feel free to create a pull request or contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.

## Reference
1. Hao-Yun Chen, Pei-Hsin Wang, Chun-Hao Liu, Shih-Chieh Chang, Jia-Yu Pan, Yu-Ting Chen, Wei Wei, and Da-Cheng Juan. Complement objective training. arXiv preprint arXiv:1903.01182, 2019.
2. Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Doll ́ar.Focal  loss  for  dense  object  detection. In Proceedings  of  the  IEEE international conference on computer vision, pages 2980–2988, 2017.
3. Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, andMu Li.  Bag of tricks for image classification with convolutional neuralnetworks.  InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 558–567, 2019.
4. https://github.com/calmisential/Basic_CNNs_TensorFlow2
5. https://github.com/Hsuxu/Loss_ToolBox-PyTorch
6. https://github.com/weiaicunzai/pytorch-cifar100
