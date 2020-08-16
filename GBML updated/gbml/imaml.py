import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
import copy
import pdb

from gbml.gbml import GBML
from utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent

class iMAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        if self.args.TTSA == True:
            self._init_opt_ttsa()
        else:
            self._init_opt()
        self.lamb = self.args.lamb
        self.n_cg = self.args.n_cg
        return None

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target):

        train_logit = fmodel(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target)
        diffopt.step(inner_loss)
        return None

    @torch.enable_grad()
    def inner_loop_reg(self, fmodel, diffopt, train_input, train_target):

        train_logit = fmodel(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target)
        inner_loss += (self.lamb / 2.) * ((torch.nn.utils.parameters_to_vector(self.network.parameters()
                                                                               ) - torch.nn.utils.parameters_to_vector(fmodel.parameters()))**2).sum()
        diffopt.step(inner_loss)
        return None

    @torch.no_grad()
    def cg(self, in_grad, outer_grad, params):
        x = outer_grad.clone().detach()
        r = outer_grad.clone().detach() - self.hv_prod(in_grad, x, params)
        p = r.clone().detach()
        for i in range(self.n_cg):
            Ap = self.hv_prod(in_grad, p, params)
            alpha = (r @ r) / (p @ Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new @ r_new) / (r @ r)
            p = r_new + beta * p
            r = r_new.clone().detach()
        return self.vec_to_grad(x)

    def vec_to_grad(self, vec):
        pointer = 0
        res = []
        for param in self.network.parameters():
            num_param = param.numel()
            res.append(vec[pointer:pointer + num_param].view_as(param).data)
            pointer += num_param
        return res

    @torch.enable_grad()
    def hv_prod(self, in_grad, x, params):
        hv = torch.autograd.grad(
            in_grad, params, retain_graph=True, grad_outputs=x)
        hv = torch.nn.utils.parameters_to_vector(hv).detach()
        # precondition with identity matrix
        return hv / self.lamb + x

    def outer_loop(self, batch, is_train):
        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(
            batch)

        loss_log = 0
        acc_log = 0
        grad_list = []
        loss_list = []

        task_num = 0
        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):

            if self.args.TTSA == True:
                params1 = self.phi[task_num]
                params2 = self.inner_optimizer[task_num]
            else:
                params1 = self.network
                params2 = self.inner_optimizer

            with higher.innerloop_ctx(params1, params2, track_higher_grads=False) as (fmodel, diffopt):

                for step in range(self.args.n_inner):
                    if self.args.imaml_reg == True:
                        self.inner_loop_reg(
                            fmodel, diffopt, train_input, train_target)
                    else:
                        self.inner_loop(fmodel, diffopt,
                                        train_input, train_target)

                train_logit = fmodel(train_input)
                in_loss = F.cross_entropy(train_logit, train_target)

                test_logit = fmodel(test_input)
                outer_loss = F.cross_entropy(test_logit, test_target)
                loss_log += outer_loss.item() / self.batch_size
                with torch.no_grad():
                    acc_log += get_accuracy(test_logit,
                                            test_target).item() / self.batch_size

                if is_train:
                    params = list(fmodel.parameters(time=-1))
                    in_grad = torch.nn.utils.parameters_to_vector(
                        torch.autograd.grad(in_loss, params, create_graph=True))
                    outer_grad = torch.nn.utils.parameters_to_vector(
                        torch.autograd.grad(outer_loss, params))
                    implicit_grad = self.cg(in_grad, outer_grad, params)
                    grad_list.append(implicit_grad)
                    loss_list.append(outer_loss.item())
                    if self.args.TTSA == True:
                        self.phi[task_num].load_state_dict(fmodel.state_dict())

        task_num += 1
        if is_train:
            self.outer_optimizer.zero_grad()
            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.network, grad)
            self.outer_optimizer.step()

            return loss_log, acc_log, grad_log
        else:
            return loss_log, acc_log
