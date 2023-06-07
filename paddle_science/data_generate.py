import skopt
from distutils.version import LooseVersion
import numpy as np
import scipy.io

np.random.seed(2023)


def gen_dataset(bounds, N_sample, method="pseudo", ndim=1):
    if ndim == 0:
        return bounds[0] * np.ones([N_sample, 1])
    elif ndim == 1:
        if method == "uniform":
            return np.linspace(bounds[0], bounds[1], N_sample)
        else:
            X = sample(N_sample, ndim, method)
            return X * (np.array(bounds)[1] - np.array(bounds)[0]) + np.array(bounds)[0]
    elif ndim == 2:
        if method == "uniform":
            n = int(np.sqrt(N_sample))
            x0 = np.linspace(bounds[0, 0], bounds[0, 1], n)
            x1 = np.linspace(bounds[1, 0], bounds[1, 1], n)
            X = np.concatenate([np.tile(x0, 1, n), np.tile(x1, 1, n)], axis=-1)
            return X
        else:
            X = sample(N_sample, ndim, method)

            return (
                X * (np.array(bounds)[1, :] - np.array(bounds)[0, :])
                + np.array(bounds)[0, :]
            )


################################ hpinns文章中生成数据方式  ################################################
def sample(n_samples, dimension, sampler="pseudo"):
    """Generate random or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (string): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Hammersley" (Hammersley
            sequence), or "Sobol" (Sobol sequence).
    """
    if sampler == "pseudo":
        return pseudo(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    raise ValueError("f{sampler} sampler is not available.")


def pseudo(n_samples, dimension):
    """Pseudo random."""
    # If random seed is set, then the rng based code always returns the same random
    # number, which may not be what we expect.
    # rng = np.random.default_rng(config.random_seed)
    # return rng.random(size=(n_samples, dimension), dtype=config.real(np))
    return np.random.random(size=(n_samples, dimension)).astype(dtype=np.float32)


def quasirandom(n_samples, dimension, sampler):
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            space = [(0.0, 1.0)] * dimension
            return np.array(
                sampler.generate(space, n_samples + 2)[2:], dtype=np.float32
            )
    space = [(0.0, 1.0)] * dimension
    return np.array(sampler.generate(space, n_samples), dtype=np.float32)


################################ gen dataset and save in .mat file ################################################
if __name__ == "__main__":
    # set params
    BOX = np.array([[-2, -2], [2, 3]])
    DPML = 1
    OMEGA = 2 * np.pi
    SIGMA0 = -np.log(1e-20) / (4 * DPML**3 / 3)
    l_BOX = BOX + np.array([[-DPML, -DPML], [DPML, DPML]])

    N = ((l_BOX[1] - l_BOX[0]) / 0.05).astype(int)  # 完整计算域
    g3_box = np.array([[-2, 0], [2, 3]])  # g3 计算objective的区域
    N_g3 = ((g3_box[1] - g3_box[0]) / 0.05).astype(int)

    # Training Data
    x_inner, y_inner = np.split(
        gen_dataset(l_BOX, N[0] * N[1], method="Sobol", ndim=2), 2, axis=-1
    )
    x_g3_small, y_g3_small = np.split(
        gen_dataset(g3_box, N_g3[0] * N_g3[1], method="Sobol", ndim=2), 2, axis=-1
    )

    x_boundl = np.ones([N[1], 1]) * l_BOX[0, 0]
    y_boundl = gen_dataset(l_BOX[:, 1], N[1], method="Sobol", ndim=1)

    x_boundr = np.ones([N[1], 1]) * l_BOX[1, 0]
    y_boundr = gen_dataset(l_BOX[:, 1], N[1], method="Sobol", ndim=1)

    x_boundd = gen_dataset(l_BOX[:, 0], N[0], method="Sobol", ndim=1)
    y_boundd = np.ones([N[0], 1]) * l_BOX[0, 1]

    x_boundu = gen_dataset(l_BOX[:, 0], N[0], method="Sobol", ndim=1)
    y_boundu = np.ones([N[0], 1]) * l_BOX[1, 1]

    input_train_x = np.concatenate(
        [x_g3_small, x_boundl, x_boundr, x_boundd, x_boundu, x_inner], axis=0
    ).astype(np.float32)
    input_train_y = np.concatenate(
        [y_g3_small, y_boundl, y_boundr, y_boundd, y_boundu, y_inner], axis=0
    ).astype(np.float32)
    num_opt = x_g3_small.shape[0]

    # np.savetxt(os.path.join(train_path, 'input_train.txt'), input_train[num_opt:])
    # input_train = paddle.to_tensor(input_train, dtype='float32', place='gpu:0')

    # save training data
    save_path = "./data/input_train.mat"
    data_trans = {
        "x": input_train_x,
        "y": input_train_y,
        "bound": num_opt,
    }
    scipy.io.savemat(save_path, data_trans)

    # Validation data
    num_opt_valid = N_g3[0] * N_g3[1]
    # generate valid data in g3 area for calculating objective function J
    x_opt = np.linspace(g3_box[0, 0], g3_box[1, 0], N_g3[0]).astype(np.float32)[:, None]
    y_opt = np.linspace(g3_box[0, 1], g3_box[1, 1], N_g3[1]).astype(np.float32)[:, None]
    x_opt = np.tile(x_opt, (1, y_opt.shape[0]))  # Nx x Ny
    y_opt = np.tile(y_opt, (1, x_opt.shape[0])).T  # Nx x Ny

    # generate valid data in all area for calculating pde residual
    x_val = np.linspace(l_BOX[0, 0], l_BOX[1, 0], N[0]).astype(np.float32)[:, None]
    y_val = np.linspace(l_BOX[0, 1], l_BOX[1, 1], N[1]).astype(np.float32)[:, None]
    x_val = np.tile(x_val, (1, y_val.shape[0]))  # Nx x Ny
    y_val = np.tile(y_val, (1, x_val.shape[0])).T  # Nx x Ny

    # save validation data
    save_path = "./data/input_valid.mat"
    data_trans = {
        "bound": 0,
        "x_opt": x_opt.reshape([-1, 1]),
        "y_opt": y_opt.reshape([-1, 1]),
        "x_val": x_val.reshape([-1, 1]),
        "y_val": y_val.reshape([-1, 1]),
    }
    scipy.io.savemat(save_path, data_trans)
