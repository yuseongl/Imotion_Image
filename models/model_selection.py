
def get_network(config):
    """ return given network
    """

    if config == 'CNN':
        from .CNN import CNN_128
        net = CNN_128()
    elif config == 'VGG11':
        from .VGG import vgg11_bn
        net = vgg11_bn()
    elif config == 'VGG13':
        from .VGG import vgg13_bn
        net = vgg13_bn()
    elif config == 'VGG16':
        from .VGG import vgg16_bn
        net = vgg16_bn()
    elif config == 'VGG19':
        from .VGG import vgg19_bn
        net = vgg19_bn()
    elif config == 'VGG_A':
        from .VGG_mlp import VGG_A
        net = VGG_A()
    elif config == 'ResNet11':
        from .ResNet import resnet11
        net = resnet11()
    elif config == 'ResNet18':
        from .ResNet import resnet18
        net = resnet18()
    elif config == 'ResNet34':
        from .ResNet import resnet34
        net = resnet34()
    elif config == 'ResNet50':
        from .ResNet import resnet50
        net = resnet50()
    elif config == 'ResNet101':
        from .ResNet import resnet101
        net = resnet101()
    elif config == 'vit':
        from .ViT import vit
        net = vit()

    else:
        raise NotImplementedError("the network name '{}' is not supported yet".format(config))

    return net