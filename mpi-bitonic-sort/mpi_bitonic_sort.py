import os
import argparse
from typing import Iterable

import numpy as np
from mpi4py import MPI


def merge_bitonic_parts(array: Iterable[float], start_index: int, num_elements_to_sort: int, ascending: bool = True):
    if num_elements_to_sort > 1:

        direction = 1 if ascending else -1
        mid_index = num_elements_to_sort // 2

        for i in range(start_index, start_index + mid_index):
            # swap elements if condition reached
            index_left = i
            index_right = i + mid_index
            if direction * array[index_left] > direction * array[index_right]:
                array[index_left], array[index_right] = array[index_right], array[index_left]

        merge_bitonic_parts(array, start_index, mid_index, ascending)
        merge_bitonic_parts(array, start_index + mid_index,
                            mid_index, ascending)


def do_bitonic_sort(array: Iterable[float], start_index: int = 0, num_elements_to_sort: int = -1, ascending: bool = True):
    num_elements_to_sort = len(
        array) if num_elements_to_sort == -1 else num_elements_to_sort

    if num_elements_to_sort > 1:

        mid_index = num_elements_to_sort // 2

        do_bitonic_sort(array, start_index, mid_index, ascending)
        do_bitonic_sort(array, start_index + mid_index,
                        mid_index, not ascending)

        merge_bitonic_parts(array, start_index,
                            num_elements_to_sort, ascending)

def merge(array_left: Iterable[float], array_right: Iterable[float], ascending: bool = True):
    merged_data = []

    direction = 1 if ascending else -1

    i = 0
    j = 0

    while i < len(array_left) and j < len(array_right):
        elem_left = array_left[i]
        elem_right = array_right[j]
        if direction * elem_left < direction * elem_right:
            merged_data.append(elem_left)
            i += 1
        else:
            merged_data.append(elem_right)
            j += 1

    merged_data.extend(array_left[i:])
    merged_data.extend(array_right[j:])

    return merged_data

parser = argparse.ArgumentParser()
parser.add_argument('--return_time', action='store_true')
parser.add_argument('-n', action='store', default=10, type=int)
args = parser.parse_args()

RETURN_TIME = args.return_time

comm = MPI.COMM_WORLD
SIZE = comm.Get_size()

# n_processes should equal a power of 2
assert int(np.log2(SIZE)) == np.log2(SIZE)

rank = comm.Get_rank()

random_digits = np.random.rand(2**args.n)
start_time = MPI.Wtime()

random_part = comm.scatter(np.array_split(random_digits, SIZE), 0)

do_bitonic_sort(random_part)

comm.Barrier()

dimensions = int(np.log2(SIZE))

for i in range(dimensions):
    for j in range(i - 1, -1, -1):
        window = rank >> (i + 1)
        j_bit = rank >> j
        process_pair = rank ^ (1 << j)

        if window % 2 == 0 and j_bit % 2 == 0 or window % 2 == 1 and j_bit % 2 != 0:
            # print(f'compare low from {rank} to {j}')
            right_data = comm.recv(source=process_pair, tag=4)
            comm.send(random_part, dest=process_pair, tag=4)
            merged_data = merge(random_part, right_data)
            random_part = np.array_split(merged_data, 2)[0].tolist()

        else:
            # print(f'compare high from {rank} to {j}')
            comm.send(random_part, dest=process_pair, tag=4)
            right_data = comm.recv(source=process_pair, tag=4)

            merged_data = merge(random_part, right_data)
            random_part = np.array_split(merged_data, 2)[1].tolist()

comm.Barrier()

# gather all data
if rank > 0:
    # print(f'sending data from {rank} process')
    comm.send(random_part, dest=0, tag=42)
else:

    if SIZE > 1:
        left_data = random_part
        for proc in range(1, SIZE // 2):
            # print(f'receiving data from {proc} process')
            data_part = comm.recv(source=proc, tag=42)
            left_data.extend(data_part)
        
        right_data = []
        for proc in range(SIZE // 2, SIZE):
            # print(f'receiving data from {proc} process')
            data_part = comm.recv(source=proc, tag=42)
            right_data.extend(data_part)

        data = merge(left_data, right_data)
    
    else:
        data = random_part
    
    end_time = MPI.Wtime()

    if RETURN_TIME:
        with open('times', 'a') as f:
            f.write(f'{int(2**args.n)} {SIZE} {end_time - start_time:.4f}\n')

    # print(np.allclose(data, sorted(random_digits)))