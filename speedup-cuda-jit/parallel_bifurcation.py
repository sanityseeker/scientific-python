import argparse

import os
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI


def generate_next_x(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    return r * x * (1 - x)


def run_experinemt(
    r: np.ndarray,
    x_init: Optional[float] = None,
    n_last_values: int = 300
) -> list:

    n_steps = len(r)

    if x_init is None:
        x_init = np.random.rand()

    xs = np.ones(n_steps) * x_init

    res = []
    for i in range(1, n_steps):
        xs = generate_next_x(xs, r)
        if i > n_steps - n_last_values:
            res.append((r, xs))
        
    return res

parser = argparse.ArgumentParser()
parser.add_argument('--compute_graph', action='store_true')
parser.add_argument('--return_time', action='store_true')
parser.add_argument('-n_rs', action='store', default=4000, type=int)
args = parser.parse_args()

COMPUTE_GRAPH = args.compute_graph
RETURN_TIME = args.return_time

n_last_values = 300
x_exp = []

comm = MPI.COMM_WORLD
SIZE = comm.Get_size()

r_values = np.linspace(0, 4, args.n_rs)
r_parts = comm.scatter(np.array_split(r_values, SIZE), 0)

rank = comm.Get_rank()

if rank == 0:
    start_time = MPI.Wtime()

process_out = run_experinemt(r_parts, x_init=None, n_last_values=n_last_values)

comm.Barrier()
process_outputs = comm.gather(process_out, 0)

if rank == 0:
    end_time = MPI.Wtime()
    if RETURN_TIME:
        with open('times', 'a') as f:
            f.write(f'{args.n_rs} {end_time - start_time:.4f}\n')

    if COMPUTE_GRAPH:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        for n in range(SIZE):
            # print(n, len(process_outputs[n]))
            for r, exp in process_outputs[n]:
                ax.scatter(r, exp, cmap='autumn', marker='.', s=2, linewidths=0.3)

        ax.grid()
        ax.set_title(f'Bifurcation map, calc done in {end_time - start_time:.4f} sec by {SIZE} processes')
        ax.set_xlabel('r')
        ax.set_ylabel('x')
        fig.savefig(f'plot_{SIZE}.png')


        

    
