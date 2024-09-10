import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm

from crypten.mpc import MPCTensor
from crypten.mpc.ptype import ptype as Ptype
from crypten.mpc.primitives.converters import new_get_msb
from crypten.mpc.primitives.resharing import truncation

import torchvision
import torchvision.models as models
import torch.autograd.profiler as profiler

import logging
import time
import timeit
import argparse

from network import *
import torchvision
from torch.utils.data import DataLoader

from typing import Tuple

parser = argparse.ArgumentParser()
parser.add_argument("--debug", "-d", required=False, default=False, help="enabling debug-mode")
args = parser.parse_args()


def test_helper(tensor_num : int, refer, target, check, siz = 5, device = "cuda") -> Tuple[bool, int]:
    rank = comm.get().get_rank()

    tensor = torch.randn([tensor_num, siz, siz], requires_grad = False).to(device)
    repeat = 500

    flag = False
    cost = 0.0
    if tensor_num == 1:
        tensor0 = tensor[0]
        c_tensor0 = crypten.cryptensor(tensor0)
        c_tensor0_plain = c_tensor0.get_plain_text()
        if args.debug and rank == 0: print(tensor0, c_tensor0_plain)
        tic = time.perf_counter()
        for _ in range(repeat): target_val = target(c_tensor0)
        toc = time.perf_counter()
        target_val = target_val.get_plain_text()
        
        if args.debug and rank == 0: print(refer(tensor0), target_val)
        if check(refer(tensor0), target_val):
            flag = True
            cost = toc - tic
    
    return flag, cost / repeat

def compare_helper(tensor_num : int, refer, targets : list, check, siz = 100, device = "cuda") -> dict:
    # targets := list of tuple[function, str]
    res = {}
    for func, name in targets:
        correct, cost = test_helper(tensor_num, refer, func, check, siz=siz, device=device)
        res[name] = cost if correct else -cost
    return res

def show_comparision(compare_res : dict) -> None:
    for name, cost in compare_res.items():
        print(f"{name}: {cost} s.")

def test_func():
    comm.get().set_verbosity(True)
    rank = comm.get().get_rank()

    def check_tensor(x, y): 
        if hasattr(y, "_tensor"): y = y._tensor 
        return sum(sum(abs(x - y))) < 1
    
    # if rank == 0: print("Testing truncate...")

    # plain_truncate = lambda x : x / (2 ** 20)
    def antelope_truncate(x : MPCTensor) -> MPCTensor:
        trunct_down = MPCTensor(torch.ones_like(x.share._tensor).to(dtype=torch.float64, device="cuda") / (2 ** 20))
        return x * trunct_down
    
    # compare_res = compare_helper(1, plain_truncate, [(antelope_truncate, "Antelope")], check_tensor)
    # if rank == 0: show_comparision(compare_res)

    if rank == 0: print("Testing exp2...")
    from math import log

    plain_exp2 = lambda x : 2 ** x
    antelope_exp2 = lambda x : (log(2) * x).exp()
    
    def my_exp2(x : MPCTensor) -> MPCTensor:
        local_power = 2 ** x.share._tensor
        shared_power0 = MPCTensor.from_shares(local_power, src=0)
        shared_power1 = MPCTensor.from_shares(local_power, src=1)
        shared_power2 = MPCTensor.from_shares(local_power, src=2)
        return shared_power0 * shared_power1 * shared_power2


    compare_res = compare_helper(1, plain_exp2, [(antelope_exp2, "Antelope"), (my_exp2, "Keller")], check_tensor)
    if rank == 0: show_comparision(compare_res)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    import multiprocess_launcher

    launcher = multiprocess_launcher.MultiProcessLauncher(3, test_func)
    launcher.start()
    launcher.join()
    launcher.terminate()