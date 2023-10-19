import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
from tqdm import tqdm

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.featuremap import FeatureMap
from utils.avgmeter import MetricTracker
from utils.tools import get_module

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
    fm = FeatureMap(device, model)
    featuremaps, _ = fm.get_featuremaps(dataloader, get_input=get_input)
    del fm

    return featuremaps


def model_stitching(device: torch.device,
                    save_path: str,
                    model: nn.Module,
                    weights: OrderedDict,
                    featuremaps: OrderedDict,
                    dataloader: DataLoader,) -> None:
    """evaluate the additivity of the model.

    Args:
        device (torch.device): the device to run the model.
        save_path (str): the path to save the results.
        model (nn.Module): the model to extract featuremaps.
        weights (OrderedDict): the weights of the model.
        featuremaps (OrderedDict): the featuremaps of the model.
        dataloader (torch.utils.data.DataLoader): the dataloader.
    
    Return:
        None (Save the results to csv.)
    """
    logger = get_logger(f"{__name__}.model_stitching")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # initialize the tracker
    tracker = MetricTracker()

    # load the weights
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # init the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # for each layer, change the intermediate featuremaps
    for layer_name, X in featuremaps.items():
        logger.info(f"predict with W_B and H_A on {layer_name}")
        
        # get the module
        module = get_module(model, layer_name)

        # init the curr_idx
        curr_idx = 0
        def hook(module, input, output):
            nonlocal curr_idx
            output.data.copy_(X[curr_idx:curr_idx+len(output)])
            curr_idx += output.shape[0]
        handle = module.register_forward_hook(hook)
        
        # calculate the predictions of model over dataloader
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                # set the inputs to device
                inputs, labels = inputs.to(device), labels.to(device)
                # set the outputs
                outputs = model(inputs)
                # set the loss
                loss = loss_fn(outputs, labels)
                # calculate the acc
                acc = (outputs.max(1)[1] == labels).float().mean()
                # update the tracker
                tracker.update({
                    "loss": loss.item(),
                    "acc": acc.item(),
                }, n=len(inputs))
        
        # remove the hook
        handle.remove()

        # track
        tracker.track({"layer": layer_name})
    
    # save the metric
    tracker.save_to_csv(os.path.join(save_path, "model_stitching.csv"))


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
    featuremaps = get_featuremaps(device = args.device,
                                  model_path = args.modelA_path,
                                  model = model,
                                  dataloader = dataloader,
                                  get_input = False)

    # evalulate the additivity from several aspects
    logger.info("#########evalulate the model stitching....")
    model_stitching(device = args.device,
                    save_path = os.path.join(args.save_path, "model_stitching"),
                    model = model,
                    weights = torch.load(args.modelB_path, map_location="cpu"),
                    featuremaps = featuremaps,
                    dataloader = dataloader,)


if __name__ == "__main__":
    main()