from models.EAN_ResNet import ResNet as SCResNet
from models.EAN_CifarResNet import ResNet as CifarSCResNet

def get_model(cfg):
    assert cfg["model"] is not None, "model is unspecified"
    assert cfg["data"]["dataset"] is not None, "dataset is unspecified"

    model_dict = cfg["model"]
    model_name = model_dict["arch"]
    model_params = {k: v for k, v in model_dict.items() if k != "arch"}
    model = _get_model_instance(model_name)

    dataset_name = cfg["data"]["dataset"]
    num_classes = _get_num_classes(dataset_name)

    model = model(num_classes=num_classes, **model_params)
    return model


def _get_num_classes(name):
    try:
        return {
            "imagenet": 1000,
            "cifar10": 10,
            "cifar100": 100,
        }[name]
    except:
        raise ("Dataset {} not available".format(name))


def _get_model_instance(name):
    try:
        return {
            "cifarscresnet": CifarSCResNet,
            "scresnet": SCResNet,
        }[name]
    except:
        raise ("Model {} not available".format(name))
