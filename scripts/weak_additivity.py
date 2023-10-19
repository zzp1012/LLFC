import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.dissimilarity import DissimilarityMetric, DissimilarityMetricOverSamples
from utils.featuremap import FeatureMap
from utils.avgmeter import MetricTracker

def get_featuremaps(device: torch.device,
                    model_path: str,
                    model: nn.Module,
                    dataloader: DataLoader,
                    get_input: bool,) -> OrderedDict:
    """get the the featuremaps of the model.

    Args:
        device (torch.device): the device to run the model.
        model_path (str): the path of pretrained model.
        model (nn.Module): the model to extract featuremaps.
        dataloader (torch.utils.data.DataLoader): the dataloader.
        get_input (bool): whether to get the input of the model.

    Return:
        featuremaps (OrderedDict): the featuremaps of the model.
    """
    logger = get_logger(f"{__name__}.get_featuremaps")
    # get the featuremaps of model 
    logger.info(f"loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # find all the relu layers
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            layers.append(name)

    fm = FeatureMap(device, model)
    featuremaps, _ = fm.get_featuremaps(dataloader, 
        layer_names=layers, get_input=get_input)
    del fm

    return featuremaps


def eval_sparsity(save_path: str,
                  X: OrderedDict,) -> None:
    """evaluate the sparsity of the featuremaps.

    Args:
        device (torch.device): the device to run the model.
        save_path (str): the path to save the results.
        X (OrderedDict): the input featuremaps of model A and B.
    
    Return:
        None (Save the results to csv.)
    """
    logger = get_logger(f"{__name__}.eval_sparsity")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert list(X.keys()) == ["A", "B"], \
        "the keys of X should be ['A', 'B']"
    assert list(X["A"].keys()) == list(X["B"].keys()), \
        "the keys of X['A'] and X['B'] should be the same"

    # initialize the dissimilarity
    cosine_dist = DissimilarityMetric("cosine")
    cosine_dist_over_samples = DissimilarityMetricOverSamples("cosine")
    diff_dist = DissimilarityMetric("vanilla")
    diff_dist_over_samples = DissimilarityMetricOverSamples("vanilla")

    # initialize the tracker
    tracker = MetricTracker()

    # define the relu layer
    relu = nn.ReLU()

    # get the layer names
    layer_names = list(X["A"].keys())
    for layer_name in layer_names:
        logger.info(f"evaluating layer {layer_name}")

        left = relu(X["A"][layer_name]) + relu(X["B"][layer_name])
        right = relu(X["A"][layer_name] + X["B"][layer_name])
        
        # get the sparsity of the featuremaps
        cosine, coef = cosine_dist(left, right, get_coef=True)
        diff = diff_dist(left, right)

        cosine_over_samples, coef_over_samples = \
            cosine_dist_over_samples(left, right, get_coef=True)
        diff_over_samples = diff_dist_over_samples(left, right)

        # save the results
        layer_save_path = os.path.join(save_path, layer_name)
        if not os.path.exists(layer_save_path):
            os.makedirs(layer_save_path)

        torch.save(cosine_over_samples, \
            os.path.join(layer_save_path, f"cosine_over_samples.pt"))
        torch.save(coef_over_samples, \
            os.path.join(layer_save_path, f"coef_over_samples.pt"))
        torch.save(diff_over_samples, \
            os.path.join(layer_save_path, f"diff_over_samples.pt"))
        
        torch.save(X["A"][layer_name], \
            os.path.join(layer_save_path, f"sparsity_A.pt"))
        torch.save(X["B"][layer_name], \
            os.path.join(layer_save_path, f"sparsity_B.pt"))
                
        # update the tracker
        tracker.track({
            "layer": layer_name,
            "cosine": cosine.item(),
            "coef": coef.item(),
            "diff": diff.item(),
            "cosine_over_samples_mean": cosine_over_samples.mean().item(),
            "coef_over_samples_mean": coef_over_samples.mean().item(),
            "diff_over_samples_mean": diff_over_samples.mean().item(),
            "cosine_over_samples_std": cosine_over_samples.std().item(),
            "coef_over_samples_std": coef_over_samples.std().item(),
            "diff_over_samples_std": diff_over_samples.std().item(),
            "shape": X["A"][layer_name].shape,
        })

    # save the results
    tracker.save_to_csv(os.path.join(save_path, "sparsity.csv"))


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--modelA_path", default=None, type=str,
                        help='the path of pretrained model A.')
    parser.add_argument("--modelB_path", default=None, type=str,
                        help='the path of pretrained model B.')
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--model", default="vgg11", type=str,
                        help='the model name.')
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size.")
    parser.add_argument("--sample_num", default=10000, type=int,
                        help="set the sample number.")
    # set if using debug mod
    parser.add_argument("-t", "--train", action="store_true", dest="train",
                        help="enable use trainset.")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",
                         f"bs{args.bs}",
                         f"{'train' if args.train else 'test'}"])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)
    # set the seed
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)
    # save the current src
    save_current_src(save_path = args.save_path)

    # show the args.
    logger.info("#########parameters settings....")
    log_settings(args)

    # prepare the dataset
    logger.info("#########preparing dataset....")
    trainset, testset = prepare_dataset(args.dataset)
    if args.train:
        # sample a subset 
        # set the seed
        set_seed(args.seed)

        # take the random samples from the testset
        indices = torch.randperm(len(trainset))[:args.sample_num]
        dataset = Subset(trainset, indices)
    else:
        dataset = testset
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False)

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset)
    logger.info(model)

    # get the featuremaps of model A and model B
    logger.info("#########get the featuremaps....")
    X = OrderedDict()
    for key, path in zip(["A", "B"], 
                         [args.modelA_path, args.modelB_path]):
        X[key] = get_featuremaps(device = args.device,
                                 model_path = path,
                                 model = model,
                                 dataloader = dataloader,
                                 get_input=True)

    # evalulate the sparsity from several aspects
    logger.info("#########evalulate the sparsity....")
    eval_sparsity(save_path = os.path.join(args.save_path, "sparsity"),
                  X = X,)


if __name__ == "__main__":
    main()