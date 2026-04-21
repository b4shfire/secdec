#!/usr/bin/env python3

from enum import IntEnum
from multiprocessing import shared_memory
from os import environ
from os.path import exists
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import pickle
import time
import torch
import logging

ctrl_name = environ.get("QMC_CTRL_NAME", "qmc_ctrl") # must match C++
data_name = environ.get("QMC_DATA_NAME", "qmc_data") # must match C++
train_atol = float(environ.get("QMC_TRAIN_ATOL", "1e-4"))
eval_atol = float(environ.get("QMC_EVAL_ATOL", "1e-11"))
cache_filepath = environ.get("QMC_CACHE_FILEPATH", "./qmc_cache.pkl")
batch_size = int(environ.get("QMC_BATCH_SIZE", "100000"))
epoch_size = int(environ.get("QMC_EPOCH_SIZE", "100"))
num_epochs = int(environ.get("QMC_NUM_EPOCHS", "10"))
learn_rate = float(environ.get("QMC_LEARN_RATE", "5e-3"))
hidden_width_factor = int(environ.get("QMC_HIDDEN_WIDTH_FACTOR", "2"))
shmem_size = int(environ.get("QMC_SHMEM_SIZE", "160000000")) # 160MB
log_level = environ.get("QMC_LOG_LEVEL", "INFO").upper() # INFO / DEBUG

train_dtype = torch.float
eval_dtype = torch.double

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# recompilation may happen many times due to different dimensions and floats vs doubles
torch._dynamo.config.recompile_limit = 1000000
torch._dynamo.config.accumulated_recompile_limit = 1000000

# recommended setting depends on hardware
torch.set_float32_matmul_precision('high')

log = logging.getLogger("qmc_cnf")
log.setLevel(log_level)
log.propagate = False
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_handler)


# shared memory control protocol
class CtrlState(IntEnum):
    READY_TO_TRAIN = 1  # CNF ready to train
    BEGIN_TRAINING = 2  # QMC requests a training run (u64 is ndim)
    EVALS_REQUEST  = 3  # CNF requests integrand evaluations (u64 is count)
    EVALS_RESPONSE = 4  # QMC has performed the evaluations
    READY_TO_FLOW  = 5  # CNF ready to flow a batch of points
    FLOW_BATCH     = 6  # QMC requests a batch be flowed (u64 is count)


def state_transition(current_state, valid_next_states):
    valid = {int(s) for s in valid_next_states}
    ctrl.buf[0] = int(current_state)
    while ctrl.buf[0] == int(current_state):
        time.sleep(0.01)
    new_state = ctrl.buf[0]
    if new_state not in valid:
        raise RuntimeError(f"Unexpected ctrl state: got {new_state}, expected one of {valid}.")
    return CtrlState(new_state)


def set_ctrl_u64(value):
    ctrl.buf[8:16] = int(value).to_bytes(8, byteorder="little")


def get_ctrl_u64():
    return int.from_bytes(ctrl.buf[8:16], byteorder="little")


class CNF(torch.nn.Module):
    def __init__(self, ndim, hidden_width):
        super().__init__()

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
        self.forward = torch.compile(self.forward, dynamic=True)


def train_cnf(ndim):

    torch.manual_seed(0)
    hidden_width = hidden_width_factor * ndim
    flow = CNF(ndim, hidden_width).to(device, dtype=train_dtype)

    # get integrand signature (10 random evaluations) for cache key
    num_sig_points = 10
    sig_points_shared = np.ndarray((num_sig_points, ndim), np.float64, buffer=data.buf)
    np.random.default_rng(0).random(sig_points_shared.shape, out=sig_points_shared)

    set_ctrl_u64(num_sig_points)
    state_transition(CtrlState.EVALS_REQUEST, [CtrlState.EVALS_RESPONSE])

    f_shared = np.ndarray((num_sig_points), np.complex128, buffer=data.buf)
    key = f"({ndim}, {hidden_width}, {batch_size}, {epoch_size*num_epochs}, {learn_rate}, {f_shared.round(7)})"

    # try look up cached result
    if exists(cache_filepath):
        with open(cache_filepath, "rb") as f:
            cache = pickle.load(f)
            if key in cache:
                flow = flow.to(device, dtype=eval_dtype)
                flow.load_state_dict(cache[key])
                log.info("Loaded cached model")
                return flow
            else:
                log.info("No cached model found")
    else:
        log.info(f"Cache file {cache_filepath} not found; creating it.")

    # no cached result found; do training

    optim = torch.optim.Adam(flow.parameters(), lr=learn_rate)
    rng = np.random.default_rng(0)

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
        x = torch.tensor(x_shared, dtype=train_dtype, requires_grad=False).to(device)

        times[1].append(time.perf_counter())

        set_ctrl_u64(batch_size)
        state_transition(CtrlState.EVALS_REQUEST, [CtrlState.EVALS_RESPONSE])

        times[2].append(time.perf_counter())

        f_shared = np.ndarray((batch_size), np.complex128, buffer=data.buf)
        f_shared = abs(f_shared)
        f = torch.tensor(f_shared, dtype=train_dtype, requires_grad=False).to(device)

        times[3].append(time.perf_counter())

        res = odeint(flow, (x,torch.zeros([batch_size], dtype=train_dtype).to(device)), torch.tensor([1.,0.], dtype=train_dtype).to(device), atol=train_atol)
        ln_p = res[1][1]

        times[4].append(time.perf_counter())

        batch_loss = torch.mean(-f*ln_p)
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
            log.info(f"Epoch {batch // epoch_size} batch {batch % epoch_size} loss: {batch_losses[-1]}")
            log.info(f"Epoch {batch // epoch_size} batch {batch % epoch_size} ESS: {batch_ess[-1]}")

    if best_flow_state is not None:
        flow.load_state_dict(best_flow_state)
        log.info(f"Best ESS: {best_ess}")

    times = [np.array(t)[1:] for t in times]

    for i in range(1, len(times)):
        log.debug(f"[Train] Time taken for {i}: {np.sum(times[i] - times[i-1])}")

    # switch dtype for evaluation
    flow = flow.to(device, dtype=eval_dtype)

    cache = {}
    if exists(cache_filepath):
        with open(cache_filepath, "rb") as f:
            cache = pickle.load(f)
    cache[key] = flow.state_dict()
    with open(cache_filepath, "wb") as f:
        pickle.dump(cache, f)

    return flow


# takes (CNF, source points); returns (output points, output pdfs)
def forward_points(flow, x):
    with torch.no_grad():
        init_divergence = torch.zeros(x.shape[0]).to(device, dtype=eval_dtype)
        time_checkpoints = torch.tensor([0.,1.]).to(device, dtype=eval_dtype)
        res = odeint(flow, (x, init_divergence), time_checkpoints, atol=eval_atol, rtol=0)
        return res[0][1], torch.exp(-res[1][1])


def run():
    while True:  # per integrand
        ndim = get_ctrl_u64()
        log.info(f"New integrand (ndim={ndim})")

        flow = train_cnf(ndim)

        times = [[] for _ in range(4)]

        while True:  # per flow batch
            times[0].append(time.perf_counter())

            new_state = state_transition(
                CtrlState.READY_TO_FLOW, [CtrlState.BEGIN_TRAINING, CtrlState.FLOW_BATCH]
            )
            if new_state == CtrlState.BEGIN_TRAINING:
                del times[0][-1]
                times = [np.array(t)[1:] for t in times]
                for i in range(len(times)-1):
                    log.debug(f"[Eval] Time taken for {i}: {np.sum(times[i+1] - times[i])}")
                break  # new integrand

            n = get_ctrl_u64()
            x = torch.tensor(np.ndarray((n, ndim), np.float64, buffer=data.buf)).to(device)

            times[1].append(time.perf_counter())

            y, p = forward_points(flow, x)
            y = torch.clamp(y, 0, 1)

            times[2].append(time.perf_counter())

            res = np.ndarray((1, n, ndim+1), np.float64, buffer=data.buf)
            res[0,:,:] = np.concatenate((p.cpu().detach().numpy()[:,None], y.cpu().detach().numpy()[:,:]), axis=1)

            times[3].append(time.perf_counter())


def main():
    global ctrl, data

    ctrl = shared_memory.SharedMemory(ctrl_name, create=True, size=4096) # 4KiB (1 page)
    data = shared_memory.SharedMemory(data_name, create=True, size=shmem_size)
    
    try:
        log.info("Ready")
        state_transition(CtrlState.READY_TO_TRAIN, [CtrlState.BEGIN_TRAINING])
        run()
    finally:
        log.info("Exit")
        data.close()
        data.unlink()
        ctrl.close()
        ctrl.unlink()


if __name__ == "__main__":
    main()
