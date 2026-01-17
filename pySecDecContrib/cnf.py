#!/usr/bin/env python3

from multiprocessing import shared_memory
from time import sleep
import numpy as np

import torch
from torch.distributions.uniform import Uniform
from torchdiffeq import odeint

import matplotlib.pyplot as plt

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#done_already2 = False

def train_cnf(ndim, hidden_width=None, learn_rate=5e-3, batch_size=10000, epoch_size=10, num_epochs=10, cutoff_ess=0.99):
    global done_already2
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

    flow = CNF().double().to(cuda)
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
    best_ess = 0.
    best_flow_state = None

    for batch in range(num_epochs * epoch_size):
        optim.zero_grad()
        
        x = source_dist.sample([batch_size]).double().to(cuda)

        # perform korobov transform
        p = x**3 * (1-x) ** 3 # for each dimension!
        p = p.prod(dim=1)
        x = x**4 * (-20.0*x**3 + 70.0*x**2 - 84.0*x + 35.0)

        #if not done_already2:
        #    done_already2 = True
        #    plt.hist(x.cpu().detach().numpy()[:,0], bins=100)
        #    plt.savefig("x.png")

        res = odeint(flow_div_batched, (x,torch.zeros([batch_size]).to(cuda)), torch.tensor([0.,1.]).to(cuda), atol=1e-4)
        y, lnJ = res[0][1], res[1][1]
        y = torch.clamp(y, 0, 1)

        eval_points = np.ndarray((batch_size, ndim), np.float64, buffer=data.buf)
        eval_points[:,:] = y.cpu().detach().numpy()[:,:]
        #print("points[0]: ", eval_points[0])

        ctrl.buf[8:16] = batch_size.to_bytes(8, byteorder="little")
        ctrl.buf[0] = 3

        while ctrl.buf[0] == 3:
            sleep(0.01)
        
        if ctrl.buf[0] != 4:
            print("invalid state", ctrl.buf[0])
            exit(1)

        target_samples = torch.tensor(np.ndarray((batch_size), np.float64, buffer=data.buf)).to(cuda)
        #print("target_samples[5]: ", target_samples[5].cpu().detach().numpy())

        sample_losses = -lnJ - torch.log(target_samples)
        sample_losses -= torch.log(p) # add korobov transform loss
        batch_loss = torch.mean(sample_losses)
        batch_losses.append(batch_loss.cpu().detach().numpy())

        prop = torch.exp(-sample_losses) #target_samples / torch.exp(-lnJ)
        ess = torch.mean(prop)**2/torch.mean(prop**2)
        batch_ess.append(ess.cpu().detach().numpy())

        ess_val = float(ess.detach().cpu().item())
        if np.isfinite(ess_val) and ess_val > best_ess:
            best_ess = ess_val
            best_flow_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}

        if batch_ess[-1] < cutoff_ess:
            batch_loss.backward()
            optim.step()

        if (batch == 0) or ((batch+1) % epoch_size == 0) or (batch_ess[-1] >= cutoff_ess):

            print(f"Epoch {batch // epoch_size} batch {batch % epoch_size} loss: {batch_losses[-1]}")
            print(f"Epoch {batch // epoch_size} batch {batch % epoch_size} ESS: {batch_ess[-1]}")

            if batch_ess[-1] >= cutoff_ess:
                break

    if best_flow_state is not None:
        flow.load_state_dict(best_flow_state)
        print(f"Best ESS={best_ess}")

    def forward_points(x):
        with torch.no_grad():
            res = odeint(flow_div_batched, (x,torch.zeros(x.shape[0]).to(cuda)), torch.tensor([0.,1.]).to(cuda), atol=1e-4)
            return res[0][1], torch.exp(-res[1][1])
    
    return forward_points, batch_losses

done_already = False

def run():

    global done_already
    if ctrl.buf[0] != 2:
        print("invalid state", ctrl.buf[0])
        exit(1)

    ndim = int.from_bytes(ctrl.buf[8:16], byteorder="little")
    print(f"ndim={ndim}")

    forward_points, batch_losses = train_cnf(ndim)

    while True:

        ctrl.buf[0] = 5
        while ctrl.buf[0] == 5:
            sleep(0.01)

        if ctrl.buf[0] == 2:
            return run() # restart
        elif ctrl.buf[0] != 6:
            print("invalid state", ctrl.buf[0])
            exit(1)

        n = int.from_bytes(ctrl.buf[8:16], byteorder="little")
        x = torch.tensor(np.ndarray((n, ndim), np.float64, buffer=data.buf)).to(cuda)
        y, p = forward_points(x)

        y = torch.clamp(y, 0, 1)

        if not done_already:
            done_already = True

            #plt.hist(x.cpu().detach().numpy()[:,0], bins=100)
            #plt.savefig("x2.png")

            plt.hist(y.cpu().detach().numpy()[:,0], bins=100)
            plt.savefig("y2.png")

        res = np.ndarray((n, ndim+1), np.float64, buffer=data.buf)
        res[:,:] = np.concatenate((p.cpu().detach().numpy()[:,None], y.cpu().detach().numpy()[:,:]), axis=1)


def main():
    global ctrl, data

    ctrl = shared_memory.SharedMemory("qmc_ctrl", create=True, size=4096) # 4KB
    data = shared_memory.SharedMemory("qmc_data", create=True, size=4194304) # 4MB

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
