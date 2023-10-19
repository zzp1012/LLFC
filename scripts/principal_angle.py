import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from scipy.linalg import subspace_angles

# import internal libs
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.tools import get_module

def cal_principal_angle(save_path: str,
                        model: nn.Module,
                        paramsA: OrderedDict,
                        paramsB: OrderedDict,
                        topk: int,) -> None:
    """calculate the principal angle between two models.

    Args:
        save_path (str): the path to save the results.
        model (nn.Module): the model.
        paramsA (OrderedDict): the parameters of modelA.
        paramsB (OrderedDict): the parameters of modelB.
        topk (int): the top k principal directions.

    Return:
        None
    """
    logger = get_logger(f"{__name__}.cal_principal_angle")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert paramsA.keys() == paramsB.keys(), \
        "the keys of paramsA and paramsB are not equal."

    for name, paramA in paramsA.items():
        if name.endswith(".weight"):
            layer = name.split(".weight")[0]
            module = get_module(model, layer)
            if not isinstance(module, nn.Conv2d) and not isinstance(module, nn.Linear):
                continue

            paramB = paramsB[name]
            assert paramA.shape == paramB.shape, \
                f"the shape of {name} is not equal."

            # reshape the weight matrix
            if len(paramA.shape) == 4 :
                O, I, H, W = paramA.shape
                paramA_mat = paramA.reshape(O, I*H*W) # (O, I*H*W)
                paramB_mat = paramB.reshape(O, I*H*W) # (O, I*H*W)
            elif len(paramA.shape) == 2:
                O, I = paramA.shape
                paramA_mat = paramA # (O, I)
                paramB_mat = paramB # (O, I)
            else:
                raise ValueError(f"the shape of {name} is not supported.")
            logger.info(f"calculate the principal angle of {name} with shape {paramA_mat.shape}")

            # SVD decomposition
            u_A, s_A, vh_A = np.linalg.svd(paramA_mat.numpy(), full_matrices=False)
            u_B, s_B, vh_B = np.linalg.svd(paramB_mat.numpy(), full_matrices=False)

            # calculate the principal angle between two subspaces, U_A and U_B
            deg_U = np.rad2deg(subspace_angles(u_A[:, :topk], u_B[:, :topk]))
            deg_U = np.sort(deg_U)[::-1] # sort the angles in descending order

            # calculate the principal angle between two subspaces, V_A and V_B
            deg_V = np.rad2deg(subspace_angles(vh_A[:topk, :].T, vh_B[:topk, :].T))
            deg_V = np.sort(deg_V)[::-1] # sort the angles in descending order

            # save the u
            layer_save_path = os.path.join(save_path, layer)
            if not os.path.exists(layer_save_path):
                os.makedirs(layer_save_path)

            np.save(os.path.join(layer_save_path, f"u_A.npy"), u_A)
            np.save(os.path.join(layer_save_path, f"u_B.npy"), u_B)

            # save the s
            np.save(os.path.join(layer_save_path, f"s_A.npy"), s_A)
            np.save(os.path.join(layer_save_path, f"s_B.npy"), s_B)

            # save the vh
            np.save(os.path.join(layer_save_path, f"vh_A.npy"), vh_A)
            np.save(os.path.join(layer_save_path, f"vh_B.npy"), vh_B)

            # save the deg
            np.save(os.path.join(layer_save_path, f"deg_U.npy"), deg_U)
            np.save(os.path.join(layer_save_path, f"deg_V.npy"), deg_V)


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
                        help='the path of pretrained modelA.')
    parser.add_argument("--modelB_path", default=None, type=str,
                        help='the path of pretrained modelB.')
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--model", default="vgg11", type=str,
                        help='the model name.')
    parser.add_argument("--topk", default=5, type=int,
                        help="the top k principal directions.")
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
                         f"{args.model}",
                         f"topk{args.topk}"])
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

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset)
    
    # calculate the principal angles between two models' weight matrix
    logger.info("#########calculate the eigenvalues of the weight matrix....")
    cal_principal_angle(os.path.join(args.save_path, "principal_angles"),
                        model, 
                        paramsA=torch.load(args.modelA_path, map_location="cpu"), 
                        paramsB=torch.load(args.modelB_path, map_location="cpu"),
                        topk=args.topk)


if __name__ == "__main__":
    main()