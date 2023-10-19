import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.avgmeter import MetricTracker
from utils.tools import interpolate_weights, evaluation

def test_connectivity(save_path: str,
                      device: torch.device,
                      model_weights: dict,
                      model_name: str,
                      dataset: Dataset,
                      dataset_name: str,
                      batch_size: int,
                      seed: int) -> None:
    """test the linear mode connectivity
    Args:
        save_path: the path to save results
        device: GPU or CPU
        model_weights: the dict containing the model A and B's weight
        model_name: the model name
        dataset: the test dataset
        dataset_name: the dataset name
        batch_size: the batch size
        seed: the seed
    
    Returns:
        None
    """
    logger = get_logger(f"{__name__}.test_connectivity")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # set the testset loader 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # init the tracker
    tracker = MetricTracker()
    
    # prepare the alpha
    alphas = np.linspace(0, 1, 30)
    for alpha in alphas:
        logger.info(f"loading model with {(1-alpha)}*weight_A + {alpha}*weight_B")
        # prepare the weights
        weights = interpolate_weights(
            model_weights["A"], model_weights["B"], (1 - alpha), alpha)
        
        # prepare the model
        model = prepare_model(model_name, dataset_name, seed)
        # set the model weights
        model.load_state_dict(weights)
        # put the model to device
        model = model.to(device)

        # evaluate the model
        test_loss, test_acc = evaluation(device, model, dataloader)

        # print the test loss and accuracy
        logger.info(f"alpha: {alpha:.2f}, \
            test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

        # update the results
        tracker.track({
            "alpha": alpha,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

    # save the results
    tracker.save_to_csv(os.path.join(save_path, "connectivity.csv"))


def add_args() -> argparse.Namespace:
    """get arguments from the program.
    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="verification of the linear mode connectivity")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument('--seed', default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--modelA_path", default=None, type=str,
                        help="the path of model A's weights.")
    parser.add_argument("--modelB_path", default=None, type=str,
                        help="the path of model B's weights.")
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--model", default="vgg16_bn", type=str,
                        help='the model name.')
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.model}",])
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
    _, testset = prepare_dataset(args.dataset)

    # prepare the model A and B
    logger.info("#########preparing model....")
    if args.modelA_path and args.modelB_path:
        model_weights = {
            "A": torch.load(args.modelA_path, map_location="cpu"), 
            "B": torch.load(args.modelB_path, map_location="cpu"),
        }
    else:
        raise ValueError(f"either {args.modelA_path} or {args.modelB_path} is None")

    # train the model
    logger.info("#########test the linear mode connectivity....")
    test_connectivity(save_path = os.path.join(args.save_path, "connectivity"),
                      device = args.device,
                      model_weights = model_weights,
                      model_name = args.model,
                      dataset = testset,
                      dataset_name = args.dataset,
                      batch_size = 128,
                      seed = args.seed,)

if __name__ == "__main__":
    main()