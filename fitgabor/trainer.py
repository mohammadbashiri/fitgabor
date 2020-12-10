import numpy as np
import torch
from torch import optim
from tqdm import trange


def trainer_fn(gabor_generator, model_neuron,
               epochs=20000, lr=1e-3,
               fixed_std=.01,
               save_rf_every_n_epoch=None,
               optimizer=optim.Adam):
    gabor_generator.apply_changes()

    optimizer = optimizer(gabor_generator.parameters(), lr=lr)

    pbar = trange(epochs, desc="Loss: {}".format(np.nan), leave=True)
    saved_rfs = []
    for epoch in pbar:
        optimizer.zero_grad()

        # generate gabor
        gabor = gabor_generator()

        if fixed_std is not None:
            gabor_std = gabor.std()
            gabor_std_constrained = fixed_std * gabor / gabor_std

        loss = -model_neuron(gabor_std_constrained)

        loss.backward()

        def closure():
            return loss

        optimizer.step(closure)

        pbar.set_description("Loss: {:.2f}".format(loss.item()))

        if save_rf_every_n_epoch is not None:
            if (epoch % save_rf_every_n_epoch) == 0:
                saved_rfs.append(gabor.squeeze().cpu().data.numpy())

    gabor_generator.eval();
    return gabor_generator, saved_rfs