import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.featuremap import FeatureMap
from utils.dissimilarity import DissimilarityMetric, DissimilarityMetricOverSamples
from utils.avgmeter import MetricTracker
from utils.tools import interpolate_weights

def get_featuremaps(device: torch.device,
                    modelA_path: str,
                    modelB_path: str,
                    model: nn.Module,
                    dataloader: DataLoader,
                    alpha: float,
                    beta: float,):
    """get the featuremaps of model A and model B

    Args:
        device (torch.device): the device to run the model.
        modelA_path (str): the path of model A.
        modelB_path (str): the path of model B.
        model (nn.Module): the model to extract featuremaps.
        dataloader (torch.utils.data.DataLoader): the dataloader.
        alpha (float): the interpolation coefficient.
        beta (float): the interpolation coefficient.

    Return:
        None
    """
    logger = get_logger(f"{__name__}.get_featuremaps")

    # prepare the weights of model A and B
    weightA, weightB = torch.load(modelA_path, map_location="cpu"),\
         torch.load(modelB_path, map_location="cpu")
    weight_alpha = interpolate_weights(weightA, weightB, alpha, beta)

    # set the layers
    model_name = model.__class__.__name__
    if model_name.startswith("MLP") or model_name.startswith("LeNet"):
        layers = [name for name, _ in model.named_modules() if name != '']
    elif model_name.startswith("VGG"):
        layers = [name for name, module in model.named_modules() \
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU))]
    else:
        layers = None

    # get the featuremaps of interpolated model
    logger.info(f"get the featuremaps of interpolated model")
    featuremaps = dict()
    for key, weight in [("A", weightA), ("B", weightB), ("alpha", weight_alpha)]:
        logger.info(f"key: {key}")
        model.load_state_dict(weight)

        # get the featuremaps
        fm = FeatureMap(device, model)
        featuremap, _ = fm.get_featuremaps(dataloader, layer_names=layers)
        del fm

        featuremaps[key] = featuremap
    return featuremaps


def eval_linearity(save_path: str,
                   featuremaps: dict,
                   alpha: float, 
                   beta: float,) -> None:
    """evaluate the linearity over different layers and alpha and beta.

    Args:
        save_path (str): the path to save the dissimilarity.
        featuremaps (dict): the featuremaps of model A and model B.
        alpha (float): the interpolation coefficient.
        beta (float): the interpolation coefficient.

    Return:
        None
    """
    logger = get_logger(f"{__name__}.eval_linearity")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # init the dissimilarity metric
    distance_fn = DissimilarityMetric("cosine")
    distance_fn_over_samples = DissimilarityMetricOverSamples("cosine")

    # initialize the MetricTracker
    tracker = MetricTracker()

    # get the layer_names
    layer_names = list(featuremaps["A"].keys())
    # calculate the dissimilarity
    for layer_name in layer_names:
        logger.info(f"layer: {layer_name}")

        # get the featuremap of model A and model B
        featuremapA = featuremaps["A"][layer_name]
        featuremapB = featuremaps["B"][layer_name]
        featuremap_alpha = featuremaps["alpha"][layer_name]
        featuremap_int = alpha * featuremapA + beta * featuremapB

        # calculate the dissimilarity
        dist_A_B, coef_A_B = \
            distance_fn(featuremapA.cpu(), featuremapB.cpu(), get_coef=True)
        dist_alpha_int, coef_alpha_int = \
            distance_fn(featuremap_alpha.cpu(), featuremap_int.cpu(), get_coef=True)
        
        # calculate the dissimilarity over samples
        dist_A_B_over_samples, coef_A_B_over_samples = \
            distance_fn_over_samples(featuremapA.cpu(), featuremapB.cpu(), get_coef=True)
        dist_alpha_int_over_samples, coef_alpha_int_over_samples = \
            distance_fn_over_samples(featuremap_alpha.cpu(), featuremap_int.cpu(), get_coef=True)
        
        # save the dissimilarity
        res_save_path = os.path.join(save_path, layer_name)
        if not os.path.exists(res_save_path):
            os.makedirs(res_save_path)
        
        torch.save(dist_A_B_over_samples, os.path.join(res_save_path, f"dist_A_B.pt"))
        torch.save(coef_A_B_over_samples, os.path.join(res_save_path, f"coef_A_B.pt"))
        torch.save(dist_alpha_int_over_samples, os.path.join(res_save_path, f"dist_alpha_int.pt"))
        torch.save(coef_alpha_int_over_samples, os.path.join(res_save_path, f"coef_alpha_int.pt"))

        tracker.track({
            "layer": layer_name,
            "dist_A_B": dist_A_B.item(),
            "coef_A_B": coef_A_B.item(),
            "dist_alpha_int": dist_alpha_int.item(),
            "coef_alpha_int": coef_alpha_int.item(),
            "dist_A_B_over_samples_mean": dist_A_B_over_samples.mean().item(),
            "coef_A_B_over_samples_mean": coef_A_B_over_samples.mean().item(),
            "dist_alpha_int_over_samples_mean": dist_alpha_int_over_samples.mean().item(),
            "coef_alpha_int_over_samples_mean": coef_alpha_int_over_samples.mean().item(),
            "dist_A_B_over_samples_std": dist_A_B_over_samples.std().item(),
            "coef_A_B_over_samples_std": coef_A_B_over_samples.std().item(),
            "dist_alpha_int_over_samples_std": dist_alpha_int_over_samples.std().item(),
            "coef_alpha_int_over_samples_std": coef_alpha_int_over_samples.std().item(),
        })

    tracker.save_to_csv(os.path.join(save_path, "sub_linearity.csv"))


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
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="set the alpha to interpolate the weights.")
    parser.add_argument("--beta", default=0.5, type=float,
                        help="set the beta to interpolate the weights.")
    parser.add_argument("--sample_num", default=10000, type=int,
                        help="set the sample number.")
    parser.add_argument("--bs", default=128, type=int,
                        help="set the batch size.")
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
                         f"alpha{args.alpha}",
                         f"beta{args.beta}",
                         f"sample_num{args.sample_num}",
                         f"bs{args.bs}",
                         f"{'train' if args.train else 'test'}",])
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
        dataset = trainset
    else:
        dataset = testset
    # take the random samples from the testset
    indices = torch.randperm(len(dataset))[:args.sample_num]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=args.bs, shuffle=False)

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset)

    # get the featuremaps
    logger.info("#########get the featuremaps....")
    featuremaps = get_featuremaps(device=args.device,
                                  modelA_path=args.modelA_path,
                                  modelB_path=args.modelB_path,
                                  model = model,
                                  dataloader=dataloader,
                                  alpha=args.alpha,
                                  beta=args.beta,)
    
    # calculate the dissimilarity
    logger.info("#########calculate the dissimilarity....")
    eval_linearity(save_path=os.path.join(args.save_path, "exp"),
                   featuremaps=featuremaps,
                   alpha=args.alpha,
                   beta=args.beta,)


if __name__ == "__main__":
    main()