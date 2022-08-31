import argparse
import tracemalloc
import os
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI


class life_grid:
    def plot_image(img: np.ndarray):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, aspect='auto', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


class grid_class:
    def __init__(self, size_x=100, size_y=100, coeff=0.5, use_glider=False):
        self.data = np.random.choice(
            [0, 1], (size_x, size_y), replace=True, p=[coeff, 1.0-coeff])
        self.size_x = size_x
        self.size_y = size_y
        self.coeff = coeff

        if use_glider:
            self.size_x = 10
            self.size_y = 10

            self.data = np.zeros((self.size_x, self.size_y))
            self.compute_glider()

        self.change_candidates = np.zeros_like(self.data)

    def compute_glider(self):
        self.data = np.zeros((self.size_x, self.size_y))

        glider = [[1, 0, 0],
                  [0, 1, 1],
                  [1, 1, 0]]

        self.data[2:5, 2:5] = glider

    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(self.data, aspect='auto', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    def count_neighbours(self, i=0, j=0):
        n_neighbours = 0
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                n_neighbours += self.data[(x + i) % self.data.shape[0],
                                          (y + j) % self.data.shape[1]]
        return n_neighbours

    def update(self, i=0, j=0):
        count = self.count_neighbours(i, j)
        change = 0

        # if dead stay dead unless exactly three alive
        # if alive stay alive only when two or three alive
        # flip state otherwise

        if (self.data[i][j] == 0) and (count == 3) or (self.data[i][j] == 1) and ((count < 2) or (count > 3)):
            self.change_candidates[i, j] = -1 if self.data[i][j] else 1
            change = 1

        return change

    def one_time_step(self):
        # go through the map, update each cell
        change_count = 0

        change_candidates = []

        for i in np.arange(self.size_x):
            for j in np.arange(self.size_y):
                change_count += self.update(i, j)

        self.data = self.data + self.change_candidates

        self.change_candidates = np.zeros_like(self.data)

        return change_count


parser = argparse.ArgumentParser()
parser.add_argument('-images_path', action='store', default='', type=str)
parser.add_argument('-cells_path', action='store', default='', type=str)
parser.add_argument('--return_time', action='store_true')
parser.add_argument('-n_iterations', action='store', default=1000, type=int)
parser.add_argument('-grid_size', action='store', default=1000, type=int)
parser.add_argument('-grid_coef', action='store', default=0.5, type=float)
parser.add_argument('--save_processes_parts', action='store_true')
args = parser.parse_args()

RETURN_TIME = args.return_time
NUM_ITERATIONS = args.n_iterations
GRID_SIZE = args.grid_size
GRID_COEF = args.grid_coef

comm = MPI.COMM_WORLD
SIZE = comm.Get_size()
rank = comm.Get_rank()

# send data for mpi processes
if rank == 0:
    start_time = MPI.Wtime()

    if args.cells_path:
        total_living = 0

    if SIZE > 1:
        step = GRID_SIZE // SIZE + 1
        border_values = list(range(0, GRID_SIZE, step)) + [GRID_SIZE]

        for proc_num in range(1, SIZE):
            borders = [border_values[proc_num], border_values[proc_num + 1]]
            shape = comm.send(borders, dest=proc_num)
            comm.send(shape, dest=proc_num)

        grid = life_grid(GRID_SIZE, border_values[1], GRID_COEF)

    else:
        grid = life_grid(GRID_SIZE, GRID_SIZE, GRID_COEF)

    if args.images_path:
        os.makedirs(args.images_path, exist_ok=True)

else:

    if args.cells_path:
        total_living = 0

    shape = comm.recv(source=0)
    grid = life_grid(GRID_SIZE, shape[1] - shape[0] + 1, GRID_COEF)

if SIZE > 1:
    prev = (rank + SIZE - 1) % SIZE
    next = (rank + 1) % SIZE

    # np.save(f'{rank}_init', grid.data)
    comm.send(grid.data[:, -1].copy().tolist(), dest=next, tag=42)
    comm.send(grid.data[:, 0].copy().tolist(), dest=prev, tag=43)
    recvbuf_left = comm.recv(source=prev, tag=42)
    recvbuf_right = comm.recv(source=next, tag=43)

    grid.data = np.hstack((np.array(recvbuf_left).reshape(-1, 1), grid.data))
    grid.data = np.hstack((grid.data, np.array(recvbuf_right).reshape(-1, 1)))

    comm.Barrier()

# np.save(f'{rank}', grid.data)


for iter in range(NUM_ITERATIONS):

    grid.one_time_step()

    if args.cells_path:
        cells = np.sum(grid.data)
        # print(rank, cells)

    if SIZE > 1:
        comm.send(grid.data[:, -1].copy().tolist(), dest=next, tag=42)
        comm.send(grid.data[:, 0].copy().tolist(), dest=prev, tag=43)

        recvbuf_left = comm.recv(source=prev, tag=42)
        recvbuf_right = comm.recv(source=next, tag=43)

        grid.data[:, 0] = np.array(recvbuf_left)
        grid.data[:, 1] = np.array(recvbuf_right)

    if args.images_path:
        np.save(os.path.join(args.images_path, f'{rank}_{iter}'), grid.data)

    if args.cells_path:
        if SIZE > 1:
            total_living = comm.reduce(cells, op=MPI.SUM, root=0)
        else:
            total_living = cells

    # print(rank, total_living)

        comm.Barrier()

    if rank == 0:
        with open(args.cells_path, 'a') as f:
            f.write(f'{iter} {total_living}\n')

if rank == 0:
    end_time = MPI.Wtime()
    if RETURN_TIME:
        with open('times', 'a') as f:
            f.write(f'{SIZE} {end_time - start_time:.4f}\n')
