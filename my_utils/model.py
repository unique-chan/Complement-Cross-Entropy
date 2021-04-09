import torchvision.models as models
import torch.nn as nn
from my_utils.my_models import resnet_for_tiny as rst
from my_utils.my_models.efficientnet import EfficientNet


def model(network_name, num_classes, pretrained=False):
    try:
        return eval(network_name + '({}, {})'.format(num_classes, pretrained))
        # return locals()[network_name](pretrained)
    except Exception as e:
        print('[Error]', e)
        exit(1)


# Insert your model function here!
# Ref.1: https://pytorch.org/docs/stable/torchvision/models.html
# Ref.2: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# Ref.3: https://stackoverflow.com/questions/63015883/pytorch-based-resnet18-achieves-low-accuracy-on-cifar100
# Ref.4: https://nannow.github.io/pytorch/2019/01/08/Pytorch-5.html


def resnet18_for_tiny(num_classes, pretrained):
    return rst.resnet18_for_tiny(num_classes=num_classes)


def resnet34_for_tiny(num_classes, pretrained):
    return rst.resnet34_for_tiny(num_classes=num_classes)


def resnet34_for_tiny_and_t_sne(num_classes, pretrained):
    return rst.resnet34_for_tiny(num_classes=num_classes, t_sne=True)


def resnet50_for_tiny(num_classes, pretrained):
    return rst.resnet50_for_tiny(num_classes=num_classes)


def resnet101_for_tiny(num_classes, pretrained):
    return rst.resnet101_for_tiny(num_classes=num_classes)


def resnet152_for_tiny(num_classes, pretrained):
    return rst.resnet152_for_tiny(num_classes=num_classes)


def resnet18(num_classes, pretrained):
    net = models.resnet18(pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def resnet34(num_classes, pretrained):
    net = models.resnet34(pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def resnet50(num_classes, pretrained):
    net = models.resnet50(pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def resnet101(num_classes, pretrained):
    net = models.resnet101(pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def resnet152(num_classes, pretrained):
    net = models.resnet152(pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def squeezenet(num_classes, pretrained):
    net = models.squeezenet1_0(pretrained)
    net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    net.num_classes = num_classes
    return net


def efficientnet_b0(num_classes, pretrained):
    net_name = 'efficientnet-b0'
    net = EfficientNet.from_pretrained(net_name) if pretrained \
        else EfficientNet.from_name(net_name)
    return net


def efficientnet_b1(num_classes, pretrained):
    net_name = 'efficientnet-b1'
    net = EfficientNet.from_pretrained(net_name) if pretrained \
        else EfficientNet.from_name(net_name)
    return net


def efficientnet_b2(num_classes, pretrained):
    net_name = 'efficientnet-b2'
    net = EfficientNet.from_pretrained(net_name) if pretrained \
        else EfficientNet.from_name(net_name)
    return net


def efficientnet_b3(num_classes, pretrained):
    net_name = 'efficientnet-b3'
    net = EfficientNet.from_pretrained(net_name) if pretrained \
        else EfficientNet.from_name(net_name)
    return net


def efficientnet_b4(num_classes, pretrained):
    net_name = 'efficientnet-b4'
    net = EfficientNet.from_pretrained(net_name) if pretrained \
        else EfficientNet.from_name(net_name)
    return net


def efficientnet_b5(num_classes, pretrained):
    net_name = 'efficientnet-b5'
    net = EfficientNet.from_pretrained(net_name) if pretrained \
        else EfficientNet.from_name(net_name)
    return net


def efficientnet_b6(num_classes, pretrained):
    net_name = 'efficientnet-b6'
    net = EfficientNet.from_pretrained(net_name) if pretrained \
        else EfficientNet.from_name(net_name)
    return net


def efficientnet_b7(num_classes, pretrained):
    net_name = 'efficientnet-b7'
    net = EfficientNet.from_pretrained(net_name) if pretrained \
        else EfficientNet.from_name(net_name)
    return net


def densenet121(num_classes, pretrained):
    net = models.densenet121(pretrained)
    net.fc = nn.Linear(net.classifier.in_features, num_classes)
    return net


def densenet161(num_classes, pretrained):
    net = models.densenet161(pretrained)
    net.fc = nn.Linear(net.classifier.in_features, num_classes)
    return net


def densenet169(num_classes, pretrained):
    net = models.densenet169(pretrained)
    net.fc = nn.Linear(net.classifier.in_features, num_classes)
    return net


def densenet201(num_classes, pretrained):
    net = models.densenet201(pretrained)
    net.fc = nn.Linear(net.classifier.in_features, num_classes)
    return net


def resnext50_32x4d(num_classes, pretrained):
    net = models.resnext50_32x4d(pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def resnext101_32x4d(num_classes, pretrained):
    net = models.resnext101_32x8d(pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net