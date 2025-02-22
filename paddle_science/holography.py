# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn.functional as F

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import config
from ppsci.utils import logger

import numpy as np
import scipy.io
from plotting import Plot


class Funcs:
    def __init__(self):
        # define params
        self.BOX = np.array([[-2, -2], [2, 3]])
        self.DPML = 1
        self.OMEGA = 2 * np.pi
        self.SIGMA0 = -np.log(1e-20) / (4 * self.DPML**3 / 3)
        self.l_BOX = self.BOX + np.array(
            [[-self.DPML, -self.DPML], [self.DPML, self.DPML]]
        )
        self.beta = 2.0
        self.mu = 2
        self.lambda_re = None
        self.lambda_im = None
        self._in = None

        # vars for plot
        self.loss_log = []  # pde, lag, opt
        self.loss_opt = 0
        self.lambda_log = []

    # utils
    def load_data_from_mat(self, path):
        return scipy.io.loadmat(path)

    # transform
    def transform_in(self, _in):
        self._in = _in
        # Periodic BC in x
        P = self.BOX[1][0] - self.BOX[0][0] + 2 * self.DPML  # 周期长度
        w = 2 * np.pi / P
        x, y = _in["x"], _in["y"]
        input_trans = {
            "x_cos": paddle.cos(w * x),
            "x_sin": paddle.sin(w * x),
            "x_cos_2": paddle.cos(2 * w * x),
            "x_sin_2": paddle.sin(2 * w * x),
            "x_cos_3": paddle.cos(3 * w * x),
            "x_sin_3": paddle.sin(3 * w * x),
            "x_cos_4": paddle.cos(4 * w * x),
            "x_sin_4": paddle.sin(4 * w * x),
            "x_cos_5": paddle.cos(5 * w * x),
            "x_sin_5": paddle.sin(5 * w * x),
            "x_cos_6": paddle.cos(6 * w * x),
            "x_sin_6": paddle.sin(6 * w * x),
            "y": y,
            "y_cos": paddle.cos(self.OMEGA * y),
            "y_sin": paddle.sin(self.OMEGA * y),
        }
        return input_trans

    def transform_out_all(self, var):
        x, y = self._in["x"], self._in["y"]

        # Zero Dirichlet BC
        a, b = self.BOX[0][1] - self.DPML, self.BOX[1][1] + self.DPML
        t = (1 - paddle.exp(a - y)) * (1 - paddle.exp(y - b))

        # Zero Dirichlet and Neumann BC
        # a, b = BOX[0][1] - DPML, BOX[1][1] + DPML
        # t = 0.01 * (a - y) ** 2 * (y - b) ** 2

        return t * var

    def transform_out_re(self, _out):
        re = _out["e_re"]
        trans_out = self.transform_out_all(re)
        return {"e_real": trans_out}

    def transform_out_im(self, _out):
        im = _out["e_im"]
        trans_out = self.transform_out_all(im)
        return {"e_imaginary": trans_out}

    def transform_out_eps(self, _out):
        eps = _out["eps"]

        # 1 <= eps <= 12
        eps = F.sigmoid(eps) * 11 + 1

        return {"epsilon": eps}

    # loss
    def init_lambda(self, output_dict, bound):
        # init lambdas of Lagrangian, only used in augmented_Lagrangian
        x, y = output_dict["x"], output_dict["y"]
        self.lambda_re = np.zeros((len(x[bound:]), 1))
        self.lambda_im = np.zeros((len(y[bound:]), 1))
        # loss_weight, PDE loss 1, PDE loss 2, Lagrangian loss 1, Lagrangian loss 2, objective loss
        self.loss_weight = [0.5 * self.mu] * 2 + [0.0, 0.0, 1.0]

    def update_lambda(self, output_dict, bound):
        loss_re, loss_im = self.get_loss_re_n_im(output_dict)
        loss_re = loss_re[bound:]
        loss_im = loss_im[bound:]
        self.lambda_re += self.mu * loss_re.detach().cpu().numpy()
        self.lambda_im += self.mu * loss_im.detach().cpu().numpy()
        self.lambda_log.append(
            [self.lambda_re.copy().squeeze(), self.lambda_im.copy().squeeze()]
        )

    def update_mu(self):
        self.mu *= self.beta
        self.loss_weight = [0.5 * self.mu] * 2 + [1, 1] + [1.0]

    def _sigma_1(self, d):
        return self.SIGMA0 * d**2 * np.heaviside(d, 0)

    def _sigma_2(self, d):
        return 2 * self.SIGMA0 * d * np.heaviside(d, 0)

    def sigma(self, x, a, b):
        """sigma(x) = 0 if a < x < b, else grows cubically from zero."""
        return self._sigma_1(a - x) + self._sigma_1(x - b)

    def dsigma(self, x, a, b):
        return -self._sigma_2(a - x) + self._sigma_2(x - b)

    def PML(self, x, y):
        x = x.numpy()
        y = y.numpy()

        sigma_x = self.sigma(x, self.BOX[0][0], self.BOX[1][0])
        AB1 = 1 / (1 + 1j / self.OMEGA * sigma_x) ** 2
        A1, B1 = AB1.real, AB1.imag

        dsigma_x = self.dsigma(x, self.BOX[0][0], self.BOX[1][0])
        AB2 = -1j / self.OMEGA * dsigma_x * AB1 / (1 + 1j / self.OMEGA * sigma_x)
        A2, B2 = AB2.real, AB2.imag

        sigma_y = self.sigma(y, self.BOX[0][1], self.BOX[1][1])
        AB3 = 1 / (1 + 1j / self.OMEGA * sigma_y) ** 2
        A3, B3 = AB3.real, AB3.imag

        dsigma_y = self.dsigma(y, self.BOX[0][1], self.BOX[1][1])
        AB4 = -1j / self.OMEGA * dsigma_y * AB3 / (1 + 1j / self.OMEGA * sigma_y)
        A4, B4 = AB4.real, AB4.imag
        return A1, B1, A2, B2, A3, B3, A4, B4

    def J(self, y):
        # Approximate the delta function
        y = y.numpy() + 1.5
        # hat function of width 2 * h
        # h = 0.5
        # return 1 / h * np.maximum(1 - np.abs(y / h), 0)
        # constant function of width 2 * h
        # h = 0.25
        # return 1 / (2 * h) * (np.abs(y) < h)
        # normal distribution of width ~2 * 2.5h
        h = 0.2
        return 1 / (h * np.pi**0.5) * np.exp(-((y / h) ** 2)) * (np.abs(y) < 0.5)

    def get_loss_re_n_im(self, output_dict):
        x, y = output_dict["x"], output_dict["y"]
        e_re = output_dict["e_real"]
        e_im = output_dict["e_imaginary"]
        eps = output_dict["epsilon"]

        condition = np.logical_and(y.numpy() < 0, y.numpy() > -1).astype(np.float32)

        eps = eps * condition + 1 - condition
        # eps = 1

        de_re_x = output_dict["de_re_x"]
        de_re_y = output_dict["de_re_y"]
        de_re_xx = output_dict["de_re_xx"]
        de_re_yy = output_dict["de_re_yy"]
        de_im_x = output_dict["de_im_x"]
        de_im_y = output_dict["de_im_y"]
        de_im_xx = output_dict["de_im_xx"]
        de_im_yy = output_dict["de_im_yy"]

        a1, b1, a2, b2, a3, b3, a4, b4 = self.PML(x, y)

        loss_re = (
            (a1 * de_re_xx + a2 * de_re_x + a3 * de_re_yy + a4 * de_re_y) / self.OMEGA
            - (b1 * de_im_xx + b2 * de_im_x + b3 * de_im_yy + b4 * de_im_y) / self.OMEGA
            + eps * self.OMEGA * e_re
        )
        loss_im = (
            (a1 * de_im_xx + a2 * de_im_x + a3 * de_im_yy + a4 * de_im_y) / self.OMEGA
            + (b1 * de_re_xx + b2 * de_re_x + b3 * de_re_yy + b4 * de_re_y) / self.OMEGA
            + eps * self.OMEGA * e_im
            + self.J(y)
        )
        return loss_re, loss_im

    def pde_loss_fun(self, output_dict):
        bound = output_dict["bound"]
        loss_re, loss_im = self.get_loss_re_n_im(output_dict)
        loss_re = loss_re[bound:]
        loss_im = loss_im[bound:]

        loss_eqs1 = F.mse_loss(
            loss_re, paddle.zeros_like(loss_re, dtype="float32"), "mean"
        )
        loss_eqs2 = F.mse_loss(
            loss_im, paddle.zeros_like(loss_im, dtype="float32"), "mean"
        )
        if self.lambda_im is None:
            self.init_lambda(output_dict, bound)
        loss_lag1 = paddle.mean(loss_re * self.lambda_re)
        loss_lag2 = paddle.mean(loss_im * self.lambda_im)

        # f1 = paddle.heaviside((x + 0.5) * (0.5 - x), paddle.to_tensor(0.5))
        # f2 = paddle.heaviside((y - 1) * (2 - y), paddle.to_tensor(0.5))
        # j = e_re[:bound] ** 2 + e_im[:bound] ** 2 - f1[:bound] * f2[:bound]
        # loss_opt_area = F.mse_loss(j, paddle.zeros_like(j, dtype="float32"), "mean")

        losses = (
            self.loss_weight[0] * loss_eqs1
            + self.loss_weight[1] * loss_eqs2
            + self.loss_weight[2] * loss_lag1
            + self.loss_weight[3] * loss_lag2
            # + self.loss_weight[4] * loss_opt_area
        )
        self.loss_log.append(float(loss_eqs1 + loss_eqs2))
        self.loss_log.append(float(loss_lag1 + loss_lag2))
        return losses  # augmented_Lagrangian

    def obj_loss_fun(self, output_dict):
        x, y = output_dict["x"], output_dict["y"]
        bound = output_dict["bound"]
        e_re = output_dict["e_real"]
        e_im = output_dict["e_imaginary"]

        f1 = paddle.heaviside((x + 0.5) * (0.5 - x), paddle.to_tensor(0.5))
        f2 = paddle.heaviside((y - 1) * (2 - y), paddle.to_tensor(0.5))
        j = e_re[:bound] ** 2 + e_im[:bound] ** 2 - f1[:bound] * f2[:bound]
        loss_opt_area = F.mse_loss(j, paddle.zeros_like(j, dtype="float32"), "mean")

        if self.lambda_im is None:
            self.init_lambda(output_dict, bound)
        losses = self.loss_weight[4] * loss_opt_area
        self.loss_log.append(float(loss_opt_area))
        self.loss_opt = float(loss_opt_area)
        return losses  # augmented_Lagrangian

    def eval_loss_fun(self, output_dict):
        x, y = output_dict["x"], output_dict["y"]
        e_re = output_dict["e_real"]
        e_im = output_dict["e_imaginary"]

        f1 = paddle.heaviside((x + 0.5) * (0.5 - x), paddle.to_tensor(0.5))
        f2 = paddle.heaviside((y - 1) * (2 - y), paddle.to_tensor(0.5))
        j = e_re**2 + e_im**2 - f1 * f2
        losses = F.mse_loss(j, paddle.zeros_like(j, dtype="float32"), "mean")

        return losses

    def eval_metric_fun(self, output_dict):
        loss_re, loss_im = self.get_loss_re_n_im(output_dict)
        eps_opt = paddle.concat([loss_re, loss_im], axis=-1)
        loss = paddle.mean(eps_opt**2)

        metric_dict = {"eval_loss": loss}
        return metric_dict


if __name__ == "__main__":
    # open FLAG for higher order differential operator when
    paddle.fluid.core.set_prim_eager_enabled(True)

    args = config.parse_args()
    ppsci.utils.misc.set_random_seed(42)
    DATASET_PATH = "./data/input_train.mat"
    DATASET_PATH_VALID = "./data/input_valid.mat"
    OUTPUT_DIR = "./output_hpinns/" if args.output_dir is None else args.output_dir

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # initialize models
    in_keys = (
        "x_cos",
        "x_sin",
        "x_cos_2",
        "x_sin_2",
        "x_cos_3",
        "x_sin_3",
        "x_cos_4",
        "x_sin_4",
        "x_cos_5",
        "x_sin_5",
        "x_cos_6",
        "x_sin_6",
        "y",
        "y_cos",
        "y_sin",
    )
    model_re = ppsci.arch.MLP(in_keys, ("e_re",), 4, 48, "tanh")
    model_im = ppsci.arch.MLP(in_keys, ("e_im",), 4, 48, "tanh")
    model_eps = ppsci.arch.MLP(in_keys, ("eps",), 4, 48, "tanh")

    # register transform
    f = Funcs()
    loss_log_obj = []

    model_re.register_input_transform(f.transform_in)
    model_im.register_input_transform(f.transform_in)
    model_eps.register_input_transform(f.transform_in)

    model_re.register_output_transform(f.transform_out_re)
    model_im.register_output_transform(f.transform_out_im)
    model_eps.register_output_transform(f.transform_out_eps)

    model_list = ppsci.arch.ModelList((model_re, model_im, model_eps))

    # set training hyper-parameters
    ITERS_PER_EPOCH = 1
    EPOCHS = 20000 if args.epochs is None else args.epochs
    MAX_ITER = 50000  # for LBFGS

    # initialize optimizer
    # Adam
    optimizer_adam = ppsci.optimizer.Adam(1e-3)((model_re, model_im, model_eps))
    # LBFGS
    optimizer_lbfgs = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)(
        (model_re, model_im, model_eps)
    )

    # maunally build constraint(s)
    label_keys = ("x", "y", "bound", "e_real", "e_imaginary", "epsilon")
    label_keys_derivative = (
        "de_re_x",
        "de_re_y",
        "de_re_xx",
        "de_re_yy",
        "de_im_x",
        "de_im_y",
        "de_im_xx",
        "de_im_yy",
    )

    train_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("x", "y", "bound"),
            "label_keys": label_keys + label_keys_derivative,
            "alias_dict": {
                "e_real": "x",
                "e_imaginary": "x",
                "epsilon": "x",
                "de_re_x": "x",
                "de_re_y": "x",
                "de_re_xx": "x",
                "de_re_yy": "x",
                "de_im_x": "x",
                "de_im_y": "x",
                "de_im_xx": "x",
                "de_im_yy": "x",
            },
        },
    }

    train_dataloader_cfg_obj = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("x", "y", "bound"),
            "label_keys": label_keys,
            "alias_dict": {"e_real": "x", "e_imaginary": "x", "epsilon": "x"},
        },
    }

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_pde,
        ppsci.loss.FunctionalLoss(f.pde_loss_fun),
        {
            "x": lambda out: out["x"],
            "y": lambda out: out["y"],
            "bound": lambda out: out["bound"],
            "e_real": lambda out: out["e_real"],
            "e_imaginary": lambda out: out["e_imaginary"],
            "epsilon": lambda out: out["epsilon"],
            "de_re_x": lambda out: jacobian(out["e_real"], out["x"]),
            "de_re_y": lambda out: jacobian(out["e_real"], out["y"]),
            "de_re_xx": lambda out: hessian(out["e_real"], out["x"]),
            "de_re_yy": lambda out: hessian(out["e_real"], out["y"]),
            "de_im_x": lambda out: jacobian(out["e_imaginary"], out["x"]),
            "de_im_y": lambda out: jacobian(out["e_imaginary"], out["y"]),
            "de_im_xx": lambda out: hessian(out["e_imaginary"], out["x"]),
            "de_im_yy": lambda out: hessian(out["e_imaginary"], out["y"]),
        },
        name="pde_sup(&lag_sup)",
    )

    sup_constraint_obj = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_obj,
        ppsci.loss.FunctionalLoss(f.obj_loss_fun),
        {key: lambda out, k=key: out[k] for key in label_keys},
        name="obj_sup",
    )

    constraint = {
        sup_constraint_pde.name: sup_constraint_pde,
        sup_constraint_obj.name: sup_constraint_obj,
    }

    # maunally build validator
    eval_dataloader_cfg_opt = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_VALID,
            "input_keys": ("x", "y", "bound"),
            "label_keys": label_keys + label_keys_derivative,
            "alias_dict": {
                "x": "x_opt",
                "y": "y_opt",
                "e_real": "x_opt",
                "e_imaginary": "x_opt",
                "epsilon": "x_opt",
                "de_re_x": "x_opt",
                "de_re_y": "x_opt",
                "de_re_xx": "x_opt",
                "de_re_yy": "x_opt",
                "de_im_x": "x_opt",
                "de_im_y": "x_opt",
                "de_im_xx": "x_opt",
                "de_im_yy": "x_opt",
            },
        },
    }

    eval_dataloader_cfg_val = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_VALID,
            "input_keys": ("x", "y", "bound"),
            "label_keys": label_keys + label_keys_derivative,
            "alias_dict": {
                "x": "x_val",
                "y": "y_val",
                "e_real": "x_val",
                "e_imaginary": "x_val",
                "epsilon": "x_val",
                "de_re_x": "x_val",
                "de_re_y": "x_val",
                "de_re_xx": "x_val",
                "de_re_yy": "x_val",
                "de_im_x": "x_val",
                "de_im_y": "x_val",
                "de_im_xx": "x_val",
                "de_im_yy": "x_val",
            },
        },
    }

    sup_validator_opt = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_opt,
        ppsci.loss.FunctionalLoss(f.eval_loss_fun),
        {
            "x": lambda out: out["x"],
            "y": lambda out: out["y"],
            "bound": lambda out: out["bound"],
            "e_real": lambda out: out["e_real"],
            "e_imaginary": lambda out: out["e_imaginary"],
            "epsilon": lambda out: out["epsilon"],
            "de_re_x": lambda out: jacobian(out["e_real"], out["x"]),
            "de_re_y": lambda out: jacobian(out["e_real"], out["y"]),
            "de_re_xx": lambda out: hessian(out["e_real"], out["x"]),
            "de_re_yy": lambda out: hessian(out["e_real"], out["y"]),
            "de_im_x": lambda out: jacobian(out["e_imaginary"], out["x"]),
            "de_im_y": lambda out: jacobian(out["e_imaginary"], out["y"]),
            "de_im_xx": lambda out: hessian(out["e_imaginary"], out["x"]),
            "de_im_yy": lambda out: hessian(out["e_imaginary"], out["y"]),
        },
        {"func": ppsci.metric.FunctionalMetric(f.eval_metric_fun)},
        name="opt_sup",
    )

    sup_validator_val = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_val,
        ppsci.loss.FunctionalLoss(f.eval_loss_fun),
        {
            "x": lambda out: out["x"],
            "y": lambda out: out["y"],
            "bound": lambda out: out["bound"],
            "e_real": lambda out: out["e_real"],
            "e_imaginary": lambda out: out["e_imaginary"],
            "epsilon": lambda out: out["epsilon"],
            "de_re_x": lambda out: jacobian(out["e_real"], out["x"]),
            "de_re_y": lambda out: jacobian(out["e_real"], out["y"]),
            "de_re_xx": lambda out: hessian(out["e_real"], out["x"]),
            "de_re_yy": lambda out: hessian(out["e_real"], out["y"]),
            "de_im_x": lambda out: jacobian(out["e_imaginary"], out["x"]),
            "de_im_y": lambda out: jacobian(out["e_imaginary"], out["y"]),
            "de_im_xx": lambda out: hessian(out["e_imaginary"], out["x"]),
            "de_im_yy": lambda out: hessian(out["e_imaginary"], out["y"]),
        },
        {"func": ppsci.metric.FunctionalMetric(f.eval_metric_fun)},
        name="val_sup",
    )

    validator = {
        sup_validator_opt.name: sup_validator_opt,
        sup_validator_val.name: sup_validator_val,
    }

    # initialize solver
    # base train
    solver = ppsci.solver.Solver(
        model_list,
        constraint,
        OUTPUT_DIR,
        optimizer_adam,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # append objective loss for plot
    loss_log_obj.append(f.loss_opt)

    # soft constraint
    solver = ppsci.solver.Solver(
        solver.model,
        constraint,
        OUTPUT_DIR,
        optimizer_lbfgs,
        None,
        1,
        ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # # penalty and augmented Lagrangian, difference between the two is updating of lambda
    # # only augmented Lagrangian: start
    # train_dict = f.load_data_from_mat(DATASET_PATH)
    # in_dict = {"x": train_dict["x"], "y": train_dict["y"]}
    # expr_dict = {
    #     "x": lambda out: out["x"],
    #     "y": lambda out: out["y"],
    #     "e_real": lambda out: out["e_real"],
    #     "e_imaginary": lambda out: out["e_imaginary"],
    #     "epsilon": lambda out: out["epsilon"],
    #     "de_re_x": lambda out: jacobian(out["e_real"], out["x"]),
    #     "de_re_y": lambda out: jacobian(out["e_real"], out["y"]),
    #     "de_re_xx": lambda out: hessian(out["e_real"], out["x"]),
    #     "de_re_yy": lambda out: hessian(out["e_real"], out["y"]),
    #     "de_im_x": lambda out: jacobian(out["e_imaginary"], out["x"]),
    #     "de_im_y": lambda out: jacobian(out["e_imaginary"], out["y"]),
    #     "de_im_xx": lambda out: hessian(out["e_imaginary"], out["x"]),
    #     "de_im_yy": lambda out: hessian(out["e_imaginary"], out["y"]),
    # }
    # f.init_lambda(in_dict, int(train_dict["bound"]))
    # f.lambda_log.append([f.lambda_re.copy().squeeze(), f.lambda_im.copy().squeeze()])
    # # only augmented Lagrangian: end

    # for i in range(1, 10):
    #     # only augmented Lagrangian: start
    #     pred_dict = solver.predict(
    #         in_dict, expr_dict, batch_size=np.shape(train_dict["x"])[0], no_grad=False
    #     )
    #     f.update_lambda(pred_dict, int(train_dict["bound"]))
    #     # only augmented Lagrangian: end

    #     f.update_mu()
    #     print(f"Iteration {i}: mu = {f.mu}\n")

    #     solver = ppsci.solver.Solver(
    #         solver.model,
    #         constraint,
    #         OUTPUT_DIR,
    #         optimizer_lbfgs,
    #         None,
    #         1,
    #         ITERS_PER_EPOCH,
    #         eval_during_train=False,
    #         validator=validator,
    #     )

    #     # train model
    #     solver.train()
    #     # evaluate after finished training
    #     solver.eval()
    #     # append objective loss for plot
    #     loss_log_obj.append(f.loss_opt)

    ################# plotting ###################
    # print("loss_log_obj:", loss_log_obj)
    # # prepare data
    # # valid data
    # N = ((f.l_BOX[1] - f.l_BOX[0]) / 0.05).astype(int)

    # valid_dict = f.load_data_from_mat(DATASET_PATH_VALID)
    # in_dict_val = {"x": valid_dict["x_val"], "y": valid_dict["y_val"]}
    # f.init_lambda(in_dict_val, int(valid_dict["bound"]))

    # pred_dict_val = solver.predict(
    #     in_dict_val,
    #     expr_dict,
    #     batch_size=np.shape(valid_dict["x_val"])[0],
    #     no_grad=False,
    # )

    # input_valid = np.stack((valid_dict["x_val"], valid_dict["y_val"]), axis=-1).reshape(
    #     [N[0], N[1], 2]
    # )
    # output_valid = np.array(
    #     [
    #         pred_dict_val["e_real"].detach().cpu().numpy(),
    #         pred_dict_val["e_imaginary"].detach().cpu().numpy(),
    #         pred_dict_val["epsilon"].detach().cpu().numpy(),
    #     ]
    # ).T.reshape([N[0], N[1], 3])

    # # train data
    # bound = int(train_dict["bound"])
    # x_train = train_dict["x"][bound:]
    # y_train = train_dict["y"][bound:]
    # input_train = np.stack((x_train, y_train), axis=-1).reshape([-1, 2])

    # log of loss
    loss_log = np.array(f.loss_log).reshape(-1, 3)

    p = Plot("test", OUTPUT_DIR)
    p.plot_6a(loss_log)
    # p.plot_6b(loss_log_obj)
    # p.plot_6c7c(f.lambda_log, input_valid, output_valid, input_train)
    # p.plot_6d(f.lambda_log)
    # p.plot_6ef(f.lambda_log)
