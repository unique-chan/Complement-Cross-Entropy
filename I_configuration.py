# hyper parameters for training
EPOCHS = 150
BATCH_SIZE = 32
NUM_CLASSES = 17
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
LEARNING_RATE = 1e-1

SAVE_CHECKPOINT_DIR = "./checkpoint/"
# SAVE_CHECKPOINT_N_ITER = 100

DATASET_DIR = "./B_dataset/"
TRAIN_DIR = DATASET_DIR + "train"
VALID_DIR = DATASET_DIR + "valid"
TEST_DIR = DATASET_DIR + "test"

TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0
# if TRAIN_SET_RATIO + TEST_SET_RATIO != 1 then
# VALID_SET_RATIO = 1 - (TRAIN_SET_RATIO + TEST_SET_RATIO).

OBJ_FUNC_INDEX = 0
# choose a objective function
# 0: Cross Entropy
# 1: COT        (Updating params once using Cross Entropy and once again with Complement Entropy)
# 2: SCOT-1     (Updating params once using Cross Entropy + γ * Complement Entropy)
# 3: SCOT-2     (Updating params once using only Complement Entropy)

# if you use OBJ_FUNC_INDEX == 2, please set the gamma.
GAMMA = 1

MODEL_INDEX = 10
# choose a network model
#     model_dic = {
#         0: torch_models.alexnet,
#         1: torch_models.vgg11,
#         2: torch_models.vgg11_bn,
#         3: torch_models.vgg13,
#         4: torch_models.vgg13_bn,
#         5: torch_models.vgg16,
#         6: torch_models.vgg16_bn,
#         7: torch_models.vgg19,
#         8: torch_models.vgg19_bn,
#         9: torch_models.resnet18,
#         10: torch_models.resnet34,
#         11: torch_models.resnet50,
#         12: torch_models.resnet101,
#         13: torch_models.resnet152,
#         14: torch_models.resnext50_32x4d,
#         15: torch_models.resnext101_32x8d,
#         16: torch_models.wide_resnet50_2,
#         17: torch_models.wide_resnet101_2,
#         18: torch_models.squeezenet1_0,
#         19: torch_models.squeezenet1_1,
#         20: torch_models.densenet121,
#         21: torch_models.densenet161,
#         22: torch_models.densenet169,
#         23: torch_models.densenet201,
#         24: torch_models.inception_v3,
#         25: torch_models.googlenet,
#         26: torch_models.shufflenet_v2_x0_5,
#         27: torch_models.shufflenet_v2_x1_0,
#         28: torch_models.shufflenet_v2_x1_5,
#         29: torch_models.shufflenet_v2_x2_0,
#         30: torch_models.mobilenet_v2,
#         31: torch_models.mnasnet0_5,
#         32: torch_models.mnasnet0_75,
#         33: torch_models.mnasnet1_0,
#         34: torch_models.mnasnet1_3,
#     }
