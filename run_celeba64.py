from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # this can be run on 2080Ti's.
    gpus = None  # [0]  # [0, 1, 2, 3]
    conf = celeba64d2c_autoenc()
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = None  # [0]  # [0, 1, 2, 3]
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = None  # [0]
    conf = celeba64d2c_autoenc_latent()
    train(conf, gpus=gpus)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    gpus = None  # [0]  # [0, 1, 2, 3]
    conf.eval_programs = ['fid(10,10)', 'fid(20,20)', 'fid(50,50)', 'fid(100,100)']
    train(conf, gpus=gpus, mode='eval')