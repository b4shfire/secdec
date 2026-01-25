#!/usr/bin/env python3

from os import environ
ctrl_name = environ.get("QMC_CTRL_NAME", "qmc_ctrl")
data_name = environ.get("QMC_DATA_NAME", "qmc_data")
atol = float(environ.get("QMC_ATOL", "nan"))

from multiprocessing import shared_memory
from time import sleep
from datetime import datetime
import numpy as np
import time
import torch
from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch._dynamo.config.recompile_limit = 1000000
torch._dynamo.config.accumulated_recompile_limit = 1000000
torch.set_float32_matmul_precision('high')

def train_cnf(ndim):

    torch.manual_seed(0)

    hidden_width = 16*ndim
    batch_size = 10000
    epoch_size = 10
    num_epochs = 10
    train_dtype=torch.float
    eval_dtype=torch.double
    
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

    flow = CNF().to(cuda, dtype=train_dtype)
    rng = np.random.default_rng(0)
    optim = torch.optim.Adam(flow.parameters(), lr=5e-3)
    
    batch_losses = []
    batch_ess = []
    best_ess = 0.
    best_flow_state = None

    times = [[] for _ in range(8)]

    for batch in range(num_epochs * epoch_size):
        optim.zero_grad()
        
        times[0].append(time.perf_counter())

        x_shared = np.ndarray((batch_size, ndim), np.float64, buffer=data.buf)
        rng.random(x_shared.shape, out=x_shared)
        x = torch.tensor(x_shared, dtype=train_dtype).to(cuda)

        times[1].append(time.perf_counter())

        ctrl.buf[8:16] = batch_size.to_bytes(8, byteorder="little")
        ctrl.buf[0] = 3
        while ctrl.buf[0] == 3:
            sleep(0.01)
        if ctrl.buf[0] != 4:
            raise ValueError(f"invalid state: {ctrl.buf[0]}")

        times[2].append(time.perf_counter())

        f_shared = np.ndarray((batch_size), np.float64, buffer=data.buf)
        f = torch.tensor(f_shared, dtype=train_dtype).to(cuda)

        times[3].append(time.perf_counter())

        res = odeint(flow, (x,torch.zeros([batch_size], dtype=train_dtype).to(cuda)), torch.tensor([1.,0.], dtype=train_dtype).to(cuda), atol=1e-2)
        ln_p = res[1][1]

        times[4].append(time.perf_counter())

        sample_losses = -f*ln_p
        batch_loss = torch.mean(sample_losses)
        batch_losses.append(batch_loss.cpu().detach().float().numpy())

        batch_ess.append(np.mean(f_shared)**2/np.mean(f_shared**2/np.exp(ln_p.cpu().detach().float().numpy())))

        if batch_ess[-1] > best_ess:
            best_ess = batch_ess[-1]
            best_flow_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}

        times[5].append(time.perf_counter())

        batch_loss.backward()

        times[6].append(time.perf_counter())

        optim.step()

        times[7].append(time.perf_counter())

        if (batch == 0) or ((batch+1) % epoch_size == 0):

            print(f"Epoch {batch // epoch_size} batch {batch % epoch_size} loss: {batch_losses[-1]}")
            print(f"Epoch {batch // epoch_size} batch {batch % epoch_size} ESS: {batch_ess[-1]}")

    if best_flow_state is not None:
        flow.load_state_dict(best_flow_state)
        print(f"Best ESS={best_ess}")

    times = [np.array(t)[1:] for t in times]

    for i in range(1, len(times)):
        print(f"[Train] Time taken for {i}: {np.sum(times[i] - times[i-1])}")

    """
    # plot loss and ESS
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("ESS")
    #ax2.set_ylim(0,1)
    x = np.arange(len(batch_losses))
    ax1.plot(x, batch_losses)
    ax2.plot(x, batch_ess, c="orange", label="ESS")
    ax2.legend()
    plt.savefig(f"loss_plots/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf")
    """

    # switch to doubles for evaluation
    flow = flow.to(cuda, dtype=eval_dtype)

    def forward_points(x):
        with torch.no_grad():
            res = odeint(flow, (x,torch.zeros(x.shape[0]).to(cuda, dtype=eval_dtype)), torch.tensor([0.,1.]).to(cuda, dtype=eval_dtype), atol=atol, rtol=0)
            return res[0][1], torch.exp(-res[1][1])

    return forward_points


def run():

    global done_already
    if ctrl.buf[0] != 2:
        print("invalid state", ctrl.buf[0])
        exit(1)

    ndim = int.from_bytes(ctrl.buf[8:16], byteorder="little")
    print(f"\nndim={ndim}")

    forward_points = train_cnf(ndim)

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

        y, p = forward_points(x)
        y = torch.clamp(y, 0, 1)

        times[2].append(time.perf_counter())

        res = np.ndarray((n, ndim+1), np.float64, buffer=data.buf)
        res[:,:] = np.concatenate((p.cpu().detach().numpy()[:,None], y.cpu().detach().numpy()[:,:]), axis=1)

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
