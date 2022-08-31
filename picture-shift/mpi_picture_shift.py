import argparse
import tracemalloc
import os
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI


def do_column_shift(img: np.ndarray, shift_size: int = 1) -> np.ndarray:
    return np.roll(img, 1, axis=1)


parser = argparse.ArgumentParser()
parser.add_argument('--compute_gif_files', action='store_true')
parser.add_argument('--return_memory', action='store_true')
parser.add_argument('--return_time', action='store_true')
parser.add_argument('-n_iterations', action='store', default=1000, type=int)
parser.add_argument('-image_path', action='store',
                    default='stripes.jpg', type=str)
parser.add_argument('--save_processes_parts', action='store_true')
args = parser.parse_args()

IMAGES_PATH = 'shifted_images_mpi'

RETURN_TIME = args.return_time
RETURN_MEMORY = args.return_memory

if RETURN_MEMORY:
    tracemalloc.start()

NUM_ITERATIONS = args.n_iterations

comm = MPI.COMM_WORLD
SIZE = comm.Get_size()

rank = comm.Get_rank()

# send data for mpi processes
if rank == 0:
    if args.image_path.split('.')[-1] == 'npy':
        img = np.load(args.image_path)
    else:
        img = plt.imread(args.image_path)

    start_time = MPI.Wtime()
    if SIZE > 1:
        step = img.shape[1] // SIZE + 1
        border_values = list(range(0, img.shape[1], step)) + [img.shape[1]]

        for proc_num in range(1, SIZE):
            data = img[:, border_values[proc_num] - 1: border_values[proc_num + 1], ...].copy()
            shape = comm.send(data.shape, dest=proc_num)
            comm.Send(data, dest=proc_num)

        data = img[:, : border_values[1], ...].copy()
    else:
        data = img.copy()

    if args.compute_gif_files:
        os.makedirs(IMAGES_PATH, exist_ok=True)

else:
    shape = comm.recv(source=0)
    data = np.empty(shape, dtype=np.uint8)
    comm.Recv(data, source=0)
    # print(rank, data.shape)

comm.Barrier()

for iter in range(NUM_ITERATIONS):
    data = do_column_shift(data)

    if SIZE > 1:
        if rank == SIZE - 1:
            send_data = data[:, 0, ...].copy()
            comm.Send(send_data, dest=0)
        elif rank == 0:
            expected_data = np.empty_like(data[:, 0, ...])
            comm.Recv(expected_data, source=SIZE - 1)
            data[:, 0, ...] = expected_data

        comm.Barrier()

        if 0 < rank < SIZE:
            expected_data = np.empty_like(data[:, 0, ...])
            comm.Recv(expected_data, source=rank - 1)
            data[:, 0, ...] = expected_data

        if rank < SIZE - 1:
            send_data = data[:, -1, ...].copy()
            comm.Send(send_data, dest=rank + 1)

        comm.Barrier()

    if iter % 20 == 0 and args.compute_gif_files:
        np.save(os.path.join(IMAGES_PATH, f'{rank}_{iter}'), data)


if args.save_processes_parts:
    np.save(f'{rank}', data)

if rank == 0:
    end_time = MPI.Wtime()

    if RETURN_TIME:
        with open('times', 'a') as f:
            f.write(f'{SIZE} {end_time - start_time:.4f}\n')

if RETURN_MEMORY:
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # print(current, peak)
    with open('memories', 'a') as f:
        f.write(f'{SIZE} {peak / 10**6:.4f}\n')
