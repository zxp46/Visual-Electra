"""
EBM training.
"""

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

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def brier_score_loss_multi(y_true, y_prob):
    """
    Brier score for multiclass.
    https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    """
    return ((y_prob - y_true) ** 2).sum(1).mean()


def main_model(args,device):
    """
    Main function.
    """
    data_sgld_dir, gen_sgld_dir, z_sgld_dir, \
    data_sgld_chain_dir, gen_sgld_chain_dir, z_sgld_chain_dir, \
    save_model_dir = utils.make_logdirs(args)

    logp_net, g = get_models(args)
    replay_buffer = get_buffer(args)

    g.train()
    g.to(device)

    # data
    train_loader, test_loader, plot = utils.get_data(args)

    batches_per_epoch = len(train_loader)
    niters = batches_per_epoch * args.n_epochs
    niters_digs = np.ceil(np.log10(niters)) + 1

    # optimization
    e_optimizer = torch.optim.Adam(g.parameters(),
                                   lr=args.lr, betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    g_optimizer = torch.optim.Adam(list(g.parameters()),
                                   lr=args.glr, betas=(args.beta1, args.beta2),
                                   weight_decay=args.weight_decay)
    scheduler_kwargs = {
        "milestones": [int(epoch * batches_per_epoch) for epoch in args.decay_epochs],
        "gamma": args.decay_rate
    }
    e_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(e_optimizer, **scheduler_kwargs)
    g_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, **scheduler_kwargs)

    itr = 0
    start_epoch = 0
    train_accs = []
    test_accs = []
    aucs = []
    briers = []
    test_lps = []
    modes = []
    kl = []
    eval_itrs = []
    if args.ckpt_path != None :
        ckpt = torch.load(args.ckpt_path)
        start_epoch = ckpt["epoch"]
        train_accs = ckpt["train_accs"]
        test_accs = ckpt["test_accs"]
        aucs = ckpt["aucs"]
        briers = ckpt["briers"]
        if "test_lps" in ckpt:
            test_lps = ckpt["test_lps"]
        else:
            test_lps = []
        modes = ckpt["modes"]
        kl = ckpt["kl"]
        eval_itrs = ckpt["eval_itrs"]
        itr = ckpt["itr"]
        g.load_state_dict(ckpt["model"]["g"])
        e_optimizer.load_state_dict(ckpt["optimizer"]["e"])
        g_optimizer.load_state_dict(ckpt["optimizer"]["g"])
        e_lr_scheduler.load_state_dict(ckpt["scheduler"]["e"])
        g_lr_scheduler.load_state_dict(ckpt["scheduler"]["g"])

    def save_ckpt(itr, overwrite=True, prefix=""):
        """
        Save a checkpoint in case job is prempted.
        """
        print(save_model_dir)
        path = os.path.join(save_model_dir, "{}_{:06d}.pt".format(prefix, itr))
        # ckpt_path will be made automatically on v2
        try:
            print("PATH",path)
            g.cpu()
            torch.save({
                # if last batch in epoch, go to next one
                "epoch": epoch + 1 if itr % batches_per_epoch == 0 else epoch,
                "train_accs": train_accs,
                "test_accs": test_accs,
                "aucs": aucs,
                "briers": briers,
                "test_lps": test_lps,
                "modes": modes,
                "kl": kl,
                "eval_itrs": eval_itrs,
                "itr": itr,
                "model": {
                    "g": g.state_dict()
                },
                "optimizer": {
                    "e": e_optimizer.state_dict(),
                    "g": g_optimizer.state_dict()
                },
                "scheduler": {
                    "e": e_lr_scheduler.state_dict(),
                    "g": g_lr_scheduler.state_dict()
                }
            }, path)
            g.to(device)
        except IOError:
            utils.print_log("Unable to save %s %d" % (path, itr), args)

    sgld_lr = 1. / args.noise_dim
    sgld_lr_z = 1. / args.noise_dim
    sgld_lr_zne = 1. / args.noise_dim

    entropy_obj = torch.tensor(0.)
    grad_ld = torch.tensor(0.)
    logq_obj = torch.tensor(0.)
    logp_obj = torch.tensor(0.)
    ld = torch.tensor(0.)
    lg_detach = torch.tensor(0.)
    ebm_gn, ent_gn = torch.tensor(0.), torch.tensor(0.)

    c_loss, train_acc, auc, brier, unsup_ent = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), \
                                                     torch.tensor(0.), torch.tensor(0.)

    t = time.time()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, args.n_epochs):
        for x_d, y_d in train_loader:
            #Load Data
            x_d = x_d.to(device)
            x_l, y_l = x_d, y_d

            x_l = x_l.to(device)
            x_l.requires_grad_()
            x_d.requires_grad_()
            y_l = y_l.to(device)

            # warmup lr
            if itr < args.warmup_iters:
                lr = args.lr * (itr + 1) / float(args.warmup_iters)
                glr = args.glr * (itr + 1) / float(args.warmup_iters)
                for param_group in e_optimizer.param_groups:
                    param_group['lr'] = lr
                for param_group in g_optimizer.param_groups:
                    param_group['lr'] = glr

            
            #Training Process
            # sample from q(x, h)
            x_g, h_g = g.sample(args.batch_size, requires_grad=True)

            # ebm (contrastive divergence) objective
            if itr % args.e_iters == 0:
                x_g_detach = x_g.detach().requires_grad_()
                print(type(x_g_detach))
                lg_detach = g.forward_d(x_g_detach).squeeze()
                #TODO return classifier logits
                ld, ld_logits = g.forward_d(x_d, return_logits=True)
                print("LD",ld.shape,"LD_logits",ld_logits.shape)
                grad_ld = torch.autograd.grad(ld.sum(), x_d,
                                                create_graph=True)[0].flatten(start_dim=1).norm(2, 1)

                logp_obj = (ld - lg_detach).mean()
                e_loss = -logp_obj + \
                            (ld ** 2).mean() * args.p_control + \
                            (lg_detach ** 2).mean() * args.n_control + \
                            (grad_ld ** 2. / 2.).mean() * args.pg_control + \
                            unsup_ent.mean() * args.clf_ent_weight
                print("E_Loss",e_loss.shape)

                #Classifier
                c_loss = torch.nn.CrossEntropyLoss()(ld_logits, y_l)

                print("C_Loss",c_loss.shape)
                chosen = ld_logits.max(1).indices
                train_acc = (chosen == y_l).float().mean().item()

                class_probs = torch.nn.functional.softmax(ld_logits.detach(), dim=1)
                if args.num_classes == 2:
                    auc = roc_auc_score(y_true=y_l.cpu(), y_score=class_probs[:, 1].cpu())
                    brier = brier_score_loss(y_true=y_l.cpu(), y_prob=class_probs[:, 1].cpu())
                else:
                    targets = torch.zeros((y_l.size(0), args.num_classes)).to(device)
                    targets.scatter_(1, y_l[:, None], 1)
                    brier = brier_score_loss_multi(y_true=targets, y_prob=class_probs).cpu()

                e_optimizer.zero_grad()
                (e_loss + args.clf_weight * c_loss).backward()
                e_optimizer.step()

            # gen obj
            if itr % args.g_iters == 0:
                lg = g.forward_d(x_g).squeeze()
                grad = torch.autograd.grad(lg.sum(), x_g, retain_graph=True)[0]
                ebm_gn = grad.norm(2, 1).mean()
                if args.ent_weight != 0.:
                    entropy_obj, ent_gn = g.entropy_obj(x_g, h_g)

                logq_obj = lg.mean() + args.ent_weight * entropy_obj

                g_loss = -logq_obj

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            # clamp sigma to (.01, max_sigma) for generators
            if args.generator_type in ["verahmc", "vera"]:
                g.clamp_sigma(args.max_sigma, sigma_min=args.min_sigma)

            # decay learning rates
            e_lr_scheduler.step()
            g_lr_scheduler.step()

            itr = itr + 1

            #Logging
            if itr % args.print_every == 0:

                # get some info to log
                new_time = time.time()
                elapsed = new_time - t
                t = new_time

                if args.generator_type == "verahmc":
                    stepsize = g.stepsize
                    post_sigma = 0
                else:
                    stepsize = 0
                    post_sigma = g.post_logsigma.exp().mean().item()

                curr_e_lr, = e_lr_scheduler.get_last_lr()
                curr_g_lr, = g_lr_scheduler.get_last_lr()

                utils.print_log("{:.1e} s/itr ({}) | "
                                "clf obj: {:.2e} ({:.4f}) ({:.4f}) ({:.4f}) | "
                                "log p obj = {:.2e}, log q obj = {:.2e}, sigma = {:.2e} | "
                                "log p(x_d) = {:.2e}, log p(x_m) = {:.2e}, ent = {:.2e} | "
                                "sgld_lr = {:.1e}, sgld_lr_z = {:.1e}, sgld_lr_zne = {:.1e} | "
                                "stepsize = {:.1e}, post_sigma = {:.1e} | "
                                "ebm gn = {:.1e}, ent gn = {:.1e} | "
                                "e-lr {:.2e}, g-lr {:.2e}".format(
                        elapsed / args.print_every, itr,
                        c_loss.item(), train_acc, auc, brier,
                        logp_obj.item(), logq_obj.item(), g.logsigma.exp().item(),
                        ld.mean().item(), lg_detach.mean().item(), entropy_obj,
                        sgld_lr, sgld_lr_z, sgld_lr_zne,
                        stepsize, post_sigma,
                        ebm_gn.item(), ent_gn.item(),
                        curr_e_lr, curr_g_lr), args)

            

            if itr % args.save_every == 0:
                save_ckpt(itr, overwrite=False)

            if itr % args.ckpt_every == 0:
                save_ckpt(itr)

            if itr % args.eval_every == 0:
                eval_itrs.append(itr)

                # evaluate the accuracy of the model at the end of the epoch on the test set
                train_accs.append(train_acc)
                accs = []
                y_ds = []
                y_preds = []
                all_class_probs = []
                g.eval()
                for x_d_, y_d_ in test_loader:
                    x_d_ = x_d_.to(device)
                    y_d_ = y_d_.to(device)

                    _, ld_logits = g.forward_d(x_d_, return_logits=True)

                    chosen = ld_logits.max(1).indices
                    acc = (chosen == y_d_).float()

                    y_preds.append(chosen)

                    class_probs = torch.nn.functional.softmax(ld_logits.detach(), dim=1)
                    y_ds.append(y_d_)
                    all_class_probs.append(class_probs)

                    accs.append(acc)

                y_ds = torch.cat(y_ds, dim=0)
                y_preds = torch.cat(y_preds, dim=0)

                all_class_probs = torch.cat(all_class_probs, dim=0)
                if args.num_classes == 2:
                    auc = roc_auc_score(y_true=y_ds.cpu(), y_score=all_class_probs[:, 1].cpu())
                aucs.append(auc)

                if args.num_classes == 2:
                    brier = brier_score_loss(y_true=y_ds.cpu(), y_prob=all_class_probs[:, 1].cpu())
                else:
                    targets = torch.zeros((y_ds.size(0), args.num_classes)).to(device)
                    targets.scatter_(1, y_ds[:, None], 1)
                    brier = brier_score_loss_multi(y_true=targets, y_prob=all_class_probs).cpu()
                briers.append(brier)

                test_accs.append(torch.cat(accs).mean().item())

                test_accs_argmax = max(enumerate(test_accs), key=itemgetter(1))[0]
                aucs_argmax = max(enumerate(aucs), key=itemgetter(1))[0]
                briers_argmin = min(enumerate(briers), key=itemgetter(1))[0]
                utils.print_log("eval itr {}, "
                                "acc {:.4f} auc {:.4f}, brier {:.4f}, "
                                "best acc {:.4f} (auc {:.4f}) (brier {:.4f}) (itr {}), "
                                "best auc {:.4f} (acc {:.4f}) (brier {:.4f}) (itr {}), "
                                "best brier {:.4f} (acc {:.4f}) (auc {:.4f}) (itr {}) ".
                                format(itr,
                                       test_accs[-1], aucs[-1], briers[-1],
                                       max(test_accs), aucs[test_accs_argmax], briers[test_accs_argmax], eval_itrs[test_accs_argmax],
                                       max(aucs), test_accs[aucs_argmax], briers[aucs_argmax], eval_itrs[aucs_argmax],
                                       min(briers), test_accs[briers_argmin], aucs[briers_argmin], eval_itrs[briers_argmin]), args)

                plt.clf()
                plt.plot(eval_itrs, train_accs, label="train")
                plt.plot(eval_itrs, test_accs, label="test")
                plt.savefig("{}/acc.png".format(args.save_dir))

                is_max = test_accs_argmax == len(test_accs) - 1

                if is_max:
                    # save model weights and plot calibration for best performing model

                    save_ckpt(itr, overwrite=True, prefix="best")

                    plt.clf()
                    if args.num_classes == 2:
                        fracpos, mean_pred = calibration_curve(y_true=y_ds.cpu(),
                                                               y_prob=all_class_probs[:, 1].cpu(), n_bins=10)
                    else:
                        fracpos, mean_pred = calibration_curve(y_true=(y_ds == y_preds).cpu(),
                                                               y_prob=all_class_probs.max(1)[0].cpu(), n_bins=10)
                    plt.plot(mean_pred, fracpos, "s-")
                    plt.xlabel("Mean predicted value")
                    plt.ylabel("Fraction of positives")
                    plt.ylim([-.05, 1.05])
                    plt.plot([0, 1], [0, 1], "k:")
                    plt.savefig("{}/cal.png".format(args.save_dir))

                g.train()



