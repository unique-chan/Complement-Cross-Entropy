import torchvision.models as torch_models
# please enumerate your own additional model(*.py)'s name on import phrase.
from b_models import resnet, squeezenet


def get_model(model_index):
    model_dic = {
        0: torch_models.alexnet,
        1: torch_models.vgg11,
        2: torch_models.vgg11_bn,
        3: torch_models.vgg13,
        4: torch_models.vgg13_bn,
        5: torch_models.vgg16,
        6: torch_models.vgg16_bn,
        7: torch_models.vgg19,
        8: torch_models.vgg19_bn,
        9: torch_models.resnet18,
        10: torch_models.resnet34,
        11: torch_models.resnet50,
        12: torch_models.resnet101,
        13: torch_models.resnet152,
        14: torch_models.resnext50_32x4d,
        15: torch_models.resnext101_32x8d,
        16: torch_models.wide_resnet50_2,
        17: torch_models.wide_resnet101_2,
        18: torch_models.squeezenet1_0,
        19: torch_models.squeezenet1_1,
        20: torch_models.densenet121,
        21: torch_models.densenet161,
        22: torch_models.densenet169,
        23: torch_models.densenet201,
        24: torch_models.inception_v3,  #
        25: torch_models.googlenet,  #
        26: torch_models.shufflenet_v2_x0_5,
        27: torch_models.shufflenet_v2_x1_0,
        28: torch_models.shufflenet_v2_x1_5,
        29: torch_models.shufflenet_v2_x2_0,
        30: torch_models.mobilenet_v2,
        31: torch_models.mnasnet0_5,
        32: torch_models.mnasnet0_75,
        33: torch_models.mnasnet1_0,
        34: torch_models.mnasnet1_3,
        # for small-scale images,
        35: resnet.ResNet18,
        36: resnet.ResNet34,
        37: resnet.ResNet50,
        38: resnet.ResNet101,
        39: resnet.ResNet152,
        40: squeezenet.squeezeNet
    }
    return model_dic[model_index]

