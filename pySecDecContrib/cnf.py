#!/usr/bin/python3

from multiprocessing import shared_memory
from time import sleep
import numpy as np

import torch
from torch.distributions.uniform import Uniform
from torchdiffeq import odeint

cuda = torch.device('cuda')

def train_cnf(ndim, hidden_width=None, learn_rate=5e-3, batch_size=10000, epoch_size=100, num_epochs=1, cutoff_ess=0.99):

    torch.manual_seed(0)
    
    if hidden_width is None:
        hidden_width = 2**(ndim+1)
    
    class CNF(torch.nn.Module):
        def __init__(self):
          super(CNF, self).__init__()
          self.fc1 = torch.nn.Linear(ndim+1, hidden_width)
          self.fc2 = torch.nn.Linear(hidden_width, ndim)
    
        def forward(self, t, x):
          y = self.fc1(torch.cat((torch.unsqueeze(t,0),x)))
          y = torch.tanh(y)
          y = self.fc2(y)
          y *= torch.sin(x * torch.pi)
          return y

    flow = CNF().to(cuda)
    flow_jac = torch.func.jacrev(flow,1)

    def flow_div(t, state):
        v = flow(t, state[0])
        div = torch.trace(flow_jac(t,state[0]))
        return (v, div)

    flow_div_batched = torch.vmap(flow_div, (None, (0, None)))
    source_dist = Uniform(torch.zeros(ndim).to(cuda), torch.ones(ndim).to(cuda))
    optim = torch.optim.Adam(flow.parameters(), lr=learn_rate)
    
    batch_losses = []
    batch_ess = []

    for batch in range(num_epochs * epoch_size):
        optim.zero_grad()
        
        x = source_dist.sample([batch_size]).to(cuda)
        res = odeint(flow_div_batched, (x,torch.zeros([batch_size]).to(cuda)), torch.tensor([0.,1.]).to(cuda), atol=1e-4)
        y, lnJ = res[0][1], res[1][1]
        
        eval_points = np.ndarray((batch_size, ndim), np.float64, buffer=data.buf)
        eval_points[:,:] = y.cpu().detach().numpy()[:,:]

        ctrl.buf[0] = 3
        ctrl.buf[8:16] = batch_size.to_bytes(8, byteorder="little")

        while ctrl.buf[0] == 3:
            sleep(0.1)

        if ctrl.buf[0] != 4:
            print("invalid state", ctrl.buf[0])
            exit(1)

        target_samples = torch.tensor(np.ndarray((batch_size), np.float64, buffer=data.buf)).to(cuda)

        sample_losses = -lnJ - torch.log(target_samples)
        batch_loss = torch.mean(sample_losses)
        batch_losses.append(batch_loss.cpu().detach().numpy())

        prop = target_samples / torch.exp(-lnJ)
        ess = torch.mean(prop)**2/torch.mean(prop**2)
        batch_ess.append(ess.cpu().detach().numpy())

        if batch_ess[-1] < cutoff_ess:
            batch_loss.backward()
            optim.step()

        if (batch == 0) or ((batch+1) % epoch_size == 0) or (batch_ess[-1] >= cutoff_ess):

            print(f"Epoch {batch // epoch_size} batch {batch % epoch_size} loss: {batch_losses[-1]}")
            print(f"Epoch {batch // epoch_size} batch {batch % epoch_size} ESS: {batch_ess[-1]}")

            if batch_ess[-1] >= cutoff_ess:
                break

    def forward_points(x):
        with torch.no_grad():
            res = odeint(flow_div_batched, (x,torch.zeros(x.shape[0]).to(cuda)), torch.tensor([0.,1.]).to(cuda), atol=1e-4)
            return res[0][1], torch.exp(-res[1][1])
    
    return forward_points, batch_losses


def run():
    if ctrl.buf[1] != 2:
        print("invalid state", ctrl.buf[1])
        exit(1)

    ndim = int.from_bytes(ctrl.buf[8:16], byteorder="little")
    print(f"ndim={ndim}")

    forward_points, batch_losses = train_cnf(ndim)

    while True:
        ctrl.buf[0] = 5
        while ctrl.buf[0] == 5:
            sleep(0.1)

        if ctrl.buf[0] == 2:
            return run() # restart
        elif ctrl.buf[0] != 6:
            print("invalid state", ctrl.buf[0])
            exit(1)

        n = int.from_bytes(ctrl.buf[8:16], byteorder="little")
        x = torch.tensor(np.ndarray((n, ndim), np.float64, buffer=data.buf)).to(cuda)
        y, p = forward_points(x)

        res = np.ndarray((n, ndim+1), np.float64, buffer=data.buf)
        res[:,:] = np.concatenate((y.cpu().detach().numpy()[:,:], p.cpu().detach().numpy()[:,None]), axis=1)


def main():
    global ctrl, data

    ctrl = shared_memory.SharedMemory("qmc_ctrl", create=True, size=4096) # 4KB
    data = shared_memory.SharedMemory("qmc_data", create=True, size=4194304) # 4MB

    ctrl.buf[0] = 1

    while ctrl.buf[0] == 1:
        sleep(0.1)

    run()

    ctrl.close()
    ctrl.unlink()

    data.close()
    data.unlink()

    print("done")


if __name__ == "__main__":
    main()
