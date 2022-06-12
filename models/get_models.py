"""
Get models based on configuration.
"""

import utils

from models import wideresnet
from models.dcgan import DCGANDiscriminator, BNDCGANDiscriminator, DCGANGenerator
from models.generator import VERAHMCGenerator, VERAGenerator
from models.jem import JEM
from models.mlp import small_mlp_ebm, large_mlp_ebm, small_mlp_generator, large_mlp_generator, MOG, NICE
from models.resnet import ResNetDiscriminator, ResNetGenerator
from models.transgan_original import Generator as TransGAN_G
from models.transgan_original import Discriminator as TransGAN_D

from utils.toy_data import TOY_DSETS
from tabular import TAB_DSETS


def get_models(args, log=True):
    """
    Get models based on configuration.
    """
    # ebm
    if args.dataset in TOY_DSETS:
        if args.mog_comps is not None:
            logp_net = MOG(args.data_dim, args.mog_comps)
        else:
            logp_net = small_mlp_ebm(args.data_dim, args.h_dim)
    elif args.dataset in TAB_DSETS:
        nout = args.num_classes if args.clf else 1
        logp_net = large_mlp_ebm(args.data_dim, nout=nout, weight_norm=False)
    elif args.dataset in ["mnist", "stackmnist"]:
        nout = 10 if args.clf else 1  # note: clf for stackmnist doesn't work
        if args.img_size is not None:
            if args.dataset == "mnist":
                logp_net = DCGANDiscriminator(in_channels=1,
                                              img_size=args.img_size,
                                              nout=nout)
            elif args.dataset == "stackmnist":
                logp_net = DCGANDiscriminator(img_size=args.img_size,
                                              nout=nout)
            else:
                raise ValueError
        else:
            if args.nice:
                logp_net = NICE(args.data_dim, 1000, 5)
            elif args.mog_comps is not None:
                logp_net = MOG(args.data_dim, args.mog_comps)
            else:
                logp_net = large_mlp_ebm(args.data_dim, nout=nout)
    elif args.dataset == "svhn" or args.dataset == "cifar10" or args.dataset == "cifar100":
        if args.dataset == "cifar100":
            nout = 100 if args.clf else 1
        else:
            nout = 10 if args.clf else 1
        norm = args.norm
        if args.resnet:
            logp_net = ResNetDiscriminator(nout=nout)
        elif args.wide_resnet:
            logp_net = wideresnet.Wide_ResNet(depth=28,
                                              widen_factor=2,
                                              num_classes=nout,
                                              norm=norm,
                                              dropout_rate=args.dropout)
        elif args.thicc_resnet:
            logp_net = wideresnet.Wide_ResNet(depth=28,
                                              widen_factor=10,
                                              num_classes=nout,
                                              norm=norm,
                                              dropout_rate=args.dropout)
        else:
            if args.norm == "batch":
                logp_net = BNDCGANDiscriminator(nout=nout)
            else:
                # TODO: modify inputs of D
                logp_net = TransGAN_D()
    else:
        raise ValueError

    # generator
    if args.generator_type in ["verahmc", "vera"]:
        # pick generator architecture based on dataset
        if args.dataset in TOY_DSETS:
            generator_net = small_mlp_generator(args.noise_dim, args.data_dim, args.h_dim)
        elif args.dataset in TAB_DSETS:
            generator_net = large_mlp_generator(args.noise_dim, args.data_dim, no_final_act=True)
        elif args.dataset in ["mnist", "stackmnist"]:
            if args.img_size is not None:
                if args.dataset == "mnist":
                    generator_net = DCGANGenerator(noise_dim=args.noise_dim,
                                                   unit_interval=args.unit_interval,
                                                   out_channels=1,
                                                   img_size=args.img_size)
                elif args.dataset == "stackmnist":
                    generator_net = DCGANGenerator(noise_dim=args.noise_dim,
                                                   unit_interval=args.unit_interval,
                                                   out_channels=3,
                                                   img_size=args.img_size)
                else:
                    raise ValueError
            else:
                generator_net = large_mlp_generator(args.noise_dim, args.data_dim, args.unit_interval, args.nice)
        elif args.dataset in ["svhn", "cifar10", "cifar100"]:
            if args.resnet:
                assert args.noise_dim == 128
                generator_net = ResNetGenerator(args.unit_interval)
            elif args.wide_resnet:
                assert args.noise_dim == 128
                generator_net = ResNetGenerator(args.unit_interval, feats=args.g_feats)
            elif args.thicc_resnet:
                assert args.noise_dim == 128
                generator_net = ResNetGenerator(args.unit_interval, feats=args.g_feats)
            else:
                generator_net = TransGAN_G(noise_dim=args.noise_dim)
        else:
            raise ValueError

        # wrap architecture with methods to sample and estimate entropy
        if args.generator_type == "verahmc":
            generator = VERAHMCGenerator(generator_net, args.noise_dim, args.mcmc_lr)
        elif args.generator_type == "vera":
            generator = VERAGenerator(generator_net, args.noise_dim, args.post_lr)
        else:
            raise ValueError

    else:
        raise ValueError

    def count_parameters(model):
        """
        Total number of model parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if log:
        utils.print_log("generator", args)
        utils.print_log(generator, args)
        utils.print_log("{} generator parameters".format(count_parameters(generator)), args)
        
    return logp_net, generator
