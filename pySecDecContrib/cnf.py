#!/usr/bin/env python3

from os import environ
from os.path import exists

from multiprocessing import shared_memory
from time import sleep
from datetime import datetime
import numpy as np
import time
import torch
from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt

import pickle

ctrl_name = environ.get("QMC_CTRL_NAME", "qmc_ctrl")
data_name = environ.get("QMC_DATA_NAME", "qmc_data")
train_only = environ.get("QMC_TRAIN_ONLY", "false").lower() == "true"
train_atol = float(environ.get("QMC_TRAIN_ATOL", "1e-4"))
eval_atol = float(environ.get("QMC_EVAL_ATOL", "1e-11"))
cache_filepath = environ.get("QMC_CACHE_FILEPATH", "./qmc_cache.pkl")
batch_size = int(environ.get("QMC_BATCH_SIZE", "10000"))
epoch_size = int(environ.get("QMC_EPOCH_SIZE", "5"))
num_epochs = int(environ.get("QMC_NUM_EPOCHS", "5"))
learn_rate = float(environ.get("QMC_LEARN_RATE", "5e-3"))
hidden_width_factor = int(environ.get("QMC_HIDDEN_WIDTH_FACTOR", "16"))

train_dtype = torch.float
eval_dtype = torch.double

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch._dynamo.config.recompile_limit = 1000000
torch._dynamo.config.accumulated_recompile_limit = 1000000
torch.set_float32_matmul_precision('high')

def train_cnf(ndim):

    torch.manual_seed(0)
    
    hidden_width = hidden_width_factor * ndim

    class CNF(torch.nn.Module):
        def __init__(self):
            super(CNF, self).__init__()

            self.fc1 = torch.nn.Linear(ndim+1, hidden_width)
            self.fc2 = torch.nn.Linear(hidden_width, ndim)

            def v(t, x):
                y = self.fc1(torch.cat((torch.unsqueeze(t,0),x)))
                y = torch.tanh(y)
                y = self.fc2(y)
                y *= torch.sin(x * torch.pi)
                return y

            jac = torch.func.jacrev(v,1)

            def v_div(t, state):
                return (v(t, state[0]), torch.trace(jac(t, state[0])))

            self.forward = torch.vmap(v_div, (None, (0, None)))
            #self.forward = torch.compile(self.forward, dynamic=True)

    flow_real = CNF().to(cuda, dtype=train_dtype)
    #flow_imag = CNF().to(cuda, dtype=train_dtype)
    
    # get function signature
    num_sig_points = 10
    sig_points_shared = np.ndarray((num_sig_points, ndim), np.float64, buffer=data.buf)
    np.random.default_rng(0).random(sig_points_shared.shape, out=sig_points_shared)

    ctrl.buf[8:16] = num_sig_points.to_bytes(8, byteorder="little")
    ctrl.buf[0] = 3
    while ctrl.buf[0] == 3:
        sleep(0.01)
    if ctrl.buf[0] != 4:
        raise ValueError(f"invalid state: {ctrl.buf[0]}")
    
    f_shared = np.ndarray((num_sig_points), np.complex128, buffer=data.buf)

    key = f"({ndim}, {hidden_width}, {batch_size}, {epoch_size*num_epochs}, {learn_rate}, {f_shared.round(7)})"

    # try look up cached result
    if exists(cache_filepath):
        with open(cache_filepath, "rb") as f:
            cache = pickle.load(f)
            if key in cache:
                flow_real = flow_real.to(cuda, dtype=eval_dtype)
                flow_real.load_state_dict(cache[key])
                print(f"Loaded cached model")
                return flow_real
            else:
                print(f"No cached models found")
    else:
        print(f"Cache file {cache_filepath} not found")
    
    # no cached result found; do training

    optim_real = torch.optim.Adam(flow_real.parameters(), lr=learn_rate)
    #optim_imag = torch.optim.Adam(flow_imag.parameters(), lr=5e-3)
    
    rng = np.random.default_rng(0)

    batch_losses_real = []
    #batch_losses_imag = []
    batch_ess_real = []
    #batch_ess_imag = []

    best_ess_real = 0.
    #best_ess_imag = 0.
    best_flow_state_real = None
    #best_flow_state_imag = None

    nofit_ess = []

    times = [[] for _ in range(8)]

    for batch in range(num_epochs * epoch_size):
        optim_real.zero_grad()
        #optim_imag.zero_grad()
    
        times[0].append(time.perf_counter())

        x_shared = np.ndarray((batch_size, ndim), np.float64, buffer=data.buf)
        rng.random(x_shared.shape, out=x_shared)
        x = torch.tensor(x_shared, dtype=train_dtype, requires_grad=False).to(cuda)

        times[1].append(time.perf_counter())

        ctrl.buf[8:16] = batch_size.to_bytes(8, byteorder="little")
        ctrl.buf[0] = 3
        while ctrl.buf[0] == 3:
            sleep(0.01)
        if ctrl.buf[0] != 4:
            raise ValueError(f"invalid state: {ctrl.buf[0]}")

        times[2].append(time.perf_counter())

        f_shared = np.ndarray((batch_size), np.complex128, buffer=data.buf)
        #f_shared.real = abs(f_shared.real)
        #f_shared.imag = abs(f_shared.imag)
        f_shared = abs(f_shared)
        #f = torch.tensor(f_shared, dtype=torch.complex128, requires_grad=False).to(cuda)
        f = torch.tensor(f_shared, dtype=train_dtype, requires_grad=False).to(cuda)

        times[3].append(time.perf_counter())

        res_real = odeint(flow_real, (x,torch.zeros([batch_size], dtype=train_dtype).to(cuda)), torch.tensor([1.,0.], dtype=train_dtype).to(cuda), atol=train_atol)
        #res_imag = odeint(flow_imag, (x,torch.zeros([batch_size], dtype=train_dtype).to(cuda)), torch.tensor([1.,0.], dtype=train_dtype).to(cuda), atol=1e-4)
        ln_p_real = res_real[1][1]
        #ln_p_imag = res_imag[1][1]

        times[4].append(time.perf_counter())

        sample_losses_real = -f.real*ln_p_real
        #sample_losses_imag = -f.imag*ln_p_imag
        batch_loss_real = torch.mean(sample_losses_real)
        #batch_loss_imag = torch.mean(sample_losses_imag)
        batch_losses_real.append(batch_loss_real.cpu().detach().float().numpy())
        #batch_losses_imag.append(batch_loss_imag.cpu().detach().float().numpy())

        # for the raw
        nofit_ess.append(np.mean(f_shared.real)**2/np.mean(f_shared.real**2))
        batch_ess_real.append(np.mean(f_shared.real)**2/np.mean(f_shared.real**2/np.exp(ln_p_real.cpu().detach().float().numpy())))
        #batch_ess_imag.append(np.mean(f_shared.imag)**2/np.mean(f_shared.imag**2/np.exp(ln_p_imag.cpu().detach().float().numpy())))

        if batch_ess_real[-1] > best_ess_real:
            best_ess_real = batch_ess_real[-1]
            best_flow_state_real = {k: v.detach().cpu().clone() for k, v in flow_real.state_dict().items()}
        #if batch_ess_imag[-1] > best_ess_imag:
        #    best_ess_imag = batch_ess_imag[-1]
        #    best_flow_state_imag = {k: v.detach().cpu().clone() for k, v in flow_imag.state_dict().items()}

        times[5].append(time.perf_counter())

        batch_loss_real.backward()
        #batch_loss_imag.backward()

        times[6].append(time.perf_counter())

        optim_real.step()
        #optim_imag.step()

        times[7].append(time.perf_counter())

        if (batch == 0) or ((batch+1) % epoch_size == 0):

            print(f"Epoch {batch // epoch_size} batch {batch % epoch_size} loss: {batch_losses_real[-1]}")# | {batch_losses_imag[-1]}")
            print(f"Epoch {batch // epoch_size} batch {batch % epoch_size} ESS: {batch_ess_real[-1]}")# | {batch_ess_imag[-1]}")

    if best_flow_state_real is not None:
        flow_real.load_state_dict(best_flow_state_real)
        #flow_imag.load_state_dict(best_flow_state_imag)
        print(f"Best ESS={best_ess_real}")# | {best_ess_imag}")

    times = [np.array(t)[1:] for t in times]

    for i in range(1, len(times)):
        print(f"[Train] Time taken for {i}: {np.sum(times[i] - times[i-1])}")

    # plot loss and ESS
    plt.clf()
    fig, ax1 = plt.subplots()
    fig.suptitle(f"ndim={ndim}, hidden_width={hidden_width}, batch_size={batch_size}, learn_rate={learn_rate}")
    ax2 = ax1.twinx()
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("ESS")
    ax2.set_ylim(0,1)
    x = np.arange(len(batch_losses_real))
    ax1.plot(x, batch_losses_real, c="C0", linewidth=0.2)#, label="Real")
    #ax1.plot(x, batch_losses_imag, c="C0", linestyle=(0,(5,5)), linewidth=0.2, label="Imag")
    ax2.plot(x, batch_ess_real, c="C1", linewidth=0.2, label="ESS")
    #ax2.plot(x, batch_ess_imag, c="C1", linestyle=(0,(5,5)), linewidth=0.2, label="ESS Imag")
    #take the mean of nofit ess and plot as horizontal line
    ax2.axhline(np.mean(np.array(nofit_ess)), c="C1", linestyle=(0,(5,5)), linewidth=0.2)
    ax2.legend()
    plt.savefig(f"loss_plots/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")
    plt.close()

    # switch to doubles for evaluation
    flow_real = flow_real.to(cuda, dtype=eval_dtype)
    #flow_imag = flow_imag.to(cuda, dtype=eval_dtype)

    cache = {}
    if exists(cache_filepath):
        with open(cache_filepath, "rb") as f:
            cache = pickle.load(f)
    cache[key] = flow_real.state_dict()
    with open(cache_filepath, "wb") as f:
        pickle.dump(cache, f)

    return flow_real


def forward_points(flow, x):
    with torch.no_grad():
        res_real = odeint(flow, (x,torch.zeros(x.shape[0]).to(cuda, dtype=eval_dtype)), torch.tensor([0.,1.]).to(cuda, dtype=eval_dtype), atol=eval_atol, rtol=0)
        #res_imag = odeint(flow_imag, (x,torch.zeros(x.shape[0]).to(cuda, dtype=eval_dtype)), torch.tensor([0.,1.]).to(cuda, dtype=eval_dtype), atol=atol, rtol=0)
        return res_real[0][1], torch.exp(-res_real[1][1])#, res_imag[0][1], torch.exp(-res_imag[1][1])

def run():

    global done_already
    if ctrl.buf[0] != 2:
        print("invalid state", ctrl.buf[0])
        exit(1)

    ndim = int.from_bytes(ctrl.buf[8:16], byteorder="little")
    print(f"\nndim={ndim}")

    flow = train_cnf(ndim)

    times = [[] for _ in range(4)]

    while True:

        times[0].append(time.perf_counter())

        ctrl.buf[0] = 5
        while ctrl.buf[0] == 5:
            sleep(0.01)
        if ctrl.buf[0] == 2:
            del times[0][-1]
            times = [np.array(t)[1:] for t in times]
            for i in range(len(times)-1):
                print(f"[Eval] Time taken for {i}: {np.sum(times[i+1] - times[i])}")
            return run() # restart
        elif ctrl.buf[0] != 6:
            print("invalid state", ctrl.buf[0])
            exit(1)

        n = int.from_bytes(ctrl.buf[8:16], byteorder="little")
        x = torch.tensor(np.ndarray((n, ndim), np.float64, buffer=data.buf)).to(cuda)

        times[1].append(time.perf_counter())

        if train_only:
            y_real = torch.ones(x.shape, dtype=torch.double)*0.5
            p_real = torch.ones(x.shape[0], dtype=torch.double)
        else:
            #y_real, p_real, y_imag, p_imag = forward_points(x)
            y_real, p_real = forward_points(flow, x)
            y_real = torch.clamp(y_real, 0, 1)
            #y_imag = torch.clamp(y_imag, 0, 1)

        times[2].append(time.perf_counter())

        res = np.ndarray((2, n, ndim+1), np.float64, buffer=data.buf)
        res[0,:,:] = np.concatenate((p_real.cpu().detach().numpy()[:,None], y_real.cpu().detach().numpy()[:,:]), axis=1)
        #res[1,:,:] = np.concatenate((p_imag.cpu().detach().numpy()[:,None], y_imag.cpu().detach().numpy()[:,:]), axis=1)

        times[3].append(time.perf_counter())

def main():
    global ctrl, data

    ctrl = shared_memory.SharedMemory(ctrl_name, create=True, size=4096) # 4KB
    data = shared_memory.SharedMemory(data_name, create=True, size=4194304) # 4MB

    ctrl.buf[0] = 1

    print("ready")

    while ctrl.buf[0] == 1:
        sleep(0.01)

    run()

    ctrl.close()
    ctrl.unlink()

    data.close()
    data.unlink()

    print("done")


if __name__ == "__main__":
    main()
