import argparse
import json
import os
import time
from operator import itemgetter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import numpy as np
import torch.utils
from torch.utils.data import DataLoader
from torch import distributions

import utils
from models.get_models import get_models
from models.jem import get_buffer, sample_q
from utils.toy_data import TOY_DSETS
from tabular import TAB_DSETS
from model import main_model
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def brier_score_loss_multi(y_true, y_prob):
    """
    Brier score for multiclass.
    https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    """
    return ((y_prob - y_true) ** 2).sum(1).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models")

    # logging
    parser.add_argument("--log_file", type=str, default="log.txt")

    # data
    parser.add_argument("--dataset", type=str, default="circles",
                        choices=list(TOY_DSETS) + list(TAB_DSETS) +
                                ["mnist", "stackmnist", "cifar10", "svhn"])
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--unit_interval", action="store_true")
    parser.add_argument("--logit", action="store_true")
    parser.add_argument("--nice", action="store_true")
    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument('--img_size', type=int)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--glr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.)
    parser.add_argument("--beta2", type=float, default=.9)
    parser.add_argument("--labels_per_class", type=int, default=0,
                        help="number of labeled examples per class, if zero then use all labels (no SSL)")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--sgld_steps", type=int, default=100)
    parser.add_argument('--mog_comps', type=int, default=None, help="Mixture of gaussians.")
    parser.add_argument("--g_feats", type=int, default=128)
    parser.add_argument("--e_iters", type=int, default=1)
    parser.add_argument("--g_iters", type=int, default=1)
    parser.add_argument("--decay_epochs", nargs="+", type=float, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--warmup_iters", type=float, default=0,
                        help="number of iterations to warmup the LR")

    # model
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--h_dim", type=int, default=100)
    parser.add_argument("--noise_dim", type=int, default=2)
    parser.add_argument("--norm", type=str, default=None, choices=[None, "batch", "group", "instance", "layer"])
    parser.add_argument("--no_g_batch_norm", action="store_true")

    parser.add_argument("--resnet", action="store_true", help="Use resnet architecture.")
    parser.add_argument("--wide_resnet", action="store_true", help="Use wide_resnet architecture")
    parser.add_argument("--thicc_resnet", action="store_true", help="Use 28-10 architecture")
    parser.add_argument("--max_sigma", type=float, default=.3)
    parser.add_argument("--min_sigma", type=float, default=.01)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--generator_type", type=str, default="vera", choices=["verahmc", "vera"])
    parser.add_argument("--clf_only", action="store_true", help="Only do classification")
    parser.add_argument("--jem", action="store_true", default=False, help="Classification and JEM training")
    parser.add_argument("--maximum_likelihood", action="store_true", default=False, help="ML baseline")
    parser.add_argument("--ssm", action="store_true", default=False, help="Sliced Score Matching baseline")

    # VAT baseline
    parser.add_argument("--vat", action="store_true", default=False, help="Run VAT instead of JEM")
    parser.add_argument("--vat_weight", type=float, default=1.0)
    parser.add_argument("--vat_eps", type=float, default=3.0)

    # JEM baseline
    parser.add_argument("--jem_baseline", action="store_true", default=False, help="Run original JEM")
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--sgld_lr", type=float, default=None)
    parser.add_argument("--sgld_std", type=float, default=.01)
    parser.add_argument("--reinit_freq", type=float, default=.05)

    # loss weighting
    parser.add_argument("--ent_weight", type=float, default=1.)
    parser.add_argument("--clf_weight", type=float, default=1.)
    parser.add_argument("--clf_ent_weight", type=float, default=0.)
    parser.add_argument("--mcmc_lr", type=float, default=.02)
    parser.add_argument("--post_lr", type=float, default=.02)

    # regularization
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--p_control", type=float, default=0.0)
    parser.add_argument("--n_control", type=float, default=0.0)
    parser.add_argument("--pg_control", type=float, default=0.0)

    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='/tmp/pgan_simp')
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_every", type=int, default=1000, help="Epochs between checkpoint save")
    parser.add_argument("--save_every", type=int, default=100000, help="Saving models for evaluation")
    parser.add_argument("--eval_every", type=int, default=200, help="Evaluating models on validation set")
    parser.add_argument("--print_every", type=int, default=10000, help="Iterations between print")
    parser.add_argument("--viz_every", type=int, default=10000, help="Iterations between visualization")
    parser.add_argument("--load_path", type=str, default=None)

    args = parser.parse_args()

    if args.img_size is not None and args.img_size not in (32, 64):
        raise ValueError

    if args.sgld_lr is None:
        args.sgld_lr = args.sgld_std ** 2. / 2.

    if args.dataset in TOY_DSETS:
        args.data_dim = 2
        args.data_size = (2, )
    elif args.dataset == "HEPMASS":
        args.data_dim = 15
        args.num_classes = 2
    elif args.dataset == "HUMAN":
        args.data_dim = 523
        args.num_classes = 6
    elif args.dataset == "CROP":
        args.data_dim = 174
        args.num_classes = 7
    elif args.dataset == "mnist":
        args.num_classes = 10
        if args.img_size:
            args.data_dim = args.img_size ** 2
            args.data_size = (1, args.img_size, args.img_size)
        else:
            args.data_dim = 784
            args.data_size = (1, 28, 28)
    elif args.dataset == "stackmnist":
        args.num_classes = 1000
        if args.img_size:
            args.data_dim = 3 * args.img_size ** 2
            args.data_size = (3, args.img_size, args.img_size)
        else:
            args.data_dim = 784 * 3
            args.data_size = (3, 28, 28)
    elif args.dataset == "svhn" or args.dataset == "cifar10":
        args.num_classes = 10
        args.data_dim = 32 * 32 * 3
        args.data_size = (3, 32, 32)
    elif args.dataset == "cifar100":
        args.num_classes = 100
        args.data_dim = 32 * 32 * 3
        args.data_size = (3, 32, 32)
    else:
        raise ValueError

    if args.dataset in TAB_DSETS:
        args.data_size = (args.data_dim, )

    assert not args.vat or args.jem, "VAT implies JEM"

    args.clf = True

    def strictly_increasing(lst):
        """
        Check if lst is strictly increasing.
        """
        return all(x < y for x, y in zip(lst[:-1], lst[1:]))

    assert strictly_increasing(args.decay_epochs), "Decay epochs should be strictly increasing"

    utils.makedirs(args.save_dir)

    torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    utils.print_log('Using {} GPU(s).'.format(torch.cuda.device_count()), args)

    with open("{}/args.txt".format(args.save_dir), 'w') as f:
        json.dump(args.__dict__, f, indent=4, sort_keys=True)

    main_model(args,device)