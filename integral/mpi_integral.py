import argparse

import os
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI


def calc_inner_func(x: float) -> float:
    return 1 / (1 + x**2)**(1 / 2)


parser = argparse.ArgumentParser()
parser.add_argument('--return_time', action='store_true')
parser.add_argument('-a', action='store', default=5, type=int)
parser.add_argument('-b', action='store', default=7, type=int)
parser.add_argument('-n_discr_points', action='store', default=1000, type=int)
args = parser.parse_args()

RETURN_TIME = args.return_time

n_points = args.n_discr_points

comm = MPI.COMM_WORLD
SIZE = comm.Get_size()
rank = comm.Get_rank()

step = n_points // SIZE
delta = (args.b - args.a) / n_points

# send starting points for mpi processes
if rank == 0:
    approximation = calc_inner_func(args.a) + calc_inner_func(args.b)
    start_values = list(range(1, n_points - step + 2, step))
    start_time = MPI.Wtime()
else:
    approximation = 0
    start_values = None

start_values = comm.bcast(start_values, root=0)
start_point = start_values[rank]

for i in range(start_point, min(n_points, start_point + step)):
    approximation += 2 * calc_inner_func(args.a + delta * i)

# print(rank, step, start_point, i, approximation)
total_sum = comm.reduce(approximation, op=MPI.SUM, root=0)

if rank == 0:
    end_time = MPI.Wtime()
    if RETURN_TIME:
        with open('times', 'a') as f:
            f.write(f'{SIZE} {end_time - start_time:.4f}\n')

    print(total_sum * delta / 2)


