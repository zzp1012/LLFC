import torch.nn as nn

# import internal libs
from utils import get_logger, set_seed

def prepare_model(model_name: str,
                  dataset: str,
                  ini_seed: int = 0) -> nn.Module:
    """prepare the random initialized model according to the name.

    Args:
        model_name (str): the model name
        dataset (str): the dataset name
        ini_seed (int): the initialization seed

    Return:
        the model
    """
    # set the seed
    set_seed(ini_seed)
    logger = get_logger(__name__)
    logger.info(f"prepare the {model_name} model for dataset {dataset}")
    if dataset == "cifar10":
        num_classes = 10
        if model_name.startswith("vgg"):
            import model.cifar_vgg as cifar_vgg
            model = cifar_vgg.__dict__[model_name](num_classes=num_classes)
        elif model_name.startswith("ResNet"):
            import model.cifar_resnet as cifar_resnet
            try:
                model = cifar_resnet.__dict__[model_name]()
            except:
                import model.cifar_resnet_plus as cifar_resnet_plus
                model = cifar_resnet_plus.__dict__[model_name]()
        else:
            raise ValueError(f"unknown model name: {model_name} for dataset {dataset}")
    elif dataset == "mnist":
        if model_name.startswith("LeNet"):
            import model.mnist_lenet as mnist_lenet
            model = mnist_lenet.__dict__[model_name]()
        elif model_name.startswith("MLP"):
            import model.mnist_mlp as mnist_mlp
            model = mnist_mlp.__dict__[model_name]()
        else:
            raise ValueError(f"unknown model name: {model_name} for dataset {dataset}")
    else:
        raise ValueError(f"{dataset} is not supported.")
    return model
