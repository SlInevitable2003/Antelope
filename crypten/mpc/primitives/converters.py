#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm
import torch
from crypten.encoder import FixedPointEncoder

from ..ptype import ptype as Ptype
from . import beaver, resharing, circuit
from .arithmetic import ArithmeticSharedTensor
from .binary import BinarySharedTensor


def _A2B(arithmetic_tensor):
    assert comm.get().get_world_size() == 3
    rank = comm.get().get_rank()

    size = arithmetic_tensor.size()
    device = arithmetic_tensor.device


    z1, z2 = BinarySharedTensor.PRZS(size, device=device).share, BinarySharedTensor.PRZS(size, device=device).share

    x1, x2 = arithmetic_tensor.share, resharing.replicate_shares(arithmetic_tensor.share)

    
    if rank == 0:
        b1 = BinarySharedTensor.from_shares(z1 ^ (x1 + x2), src=rank)
        b2 = BinarySharedTensor.from_shares(z2, src=rank)
    elif rank == 1:
        b1 = BinarySharedTensor.from_shares(z1, src=rank)
        b2 = BinarySharedTensor.from_shares(z2 ^ x1, src=rank)
    else:
        b1 = BinarySharedTensor.from_shares(z1, src=rank)
        b2 = BinarySharedTensor.from_shares(z2, src=rank)

    binary_tensor = circuit.extract_msb(b1, b2)
  
    binary_tensor.encoder = arithmetic_tensor.encoder
   

    return binary_tensor
    

def convert(tensor, ptype, **kwargs):
    tensor_name = ptype.to_tensor()
    if isinstance(tensor, tensor_name):
        return tensor
    if isinstance(tensor, ArithmeticSharedTensor) and ptype == Ptype.binary:
        return _A2B(tensor)
    else:
        raise TypeError("Cannot convert %s to %s" % (type(tensor), ptype.__name__))


def get_msb(arithmetic_tensor):
    return _A2B(arithmetic_tensor)

def three_mill(cuda_tensor, rank):
    from crypten.common.rng import generate_random_p_element
    import time
    input = cuda_tensor._tensor
    if rank==0:
        input=(abs(input)%((1<<31)-1)).int()
        input = input.reshape(1,-1).repeat(32,1)

        import torchcsprng as csprng
        urandom_gen=csprng.create_random_device_generator('/dev/urandom')
        r = torch.empty(input.shape,dtype=torch.int,device='cuda').random_(0, to=(2**31 - 1),generator=urandom_gen)
        tic = time.perf_counter()
        import encode1
        #print(input.shape[1])
        encode1.torch_launch_encode1(input,r,input.shape[1])
        toc = time.perf_counter()
        # hash
        r1 = generate_random_p_element([1], device=r.device, generator=comm.get().get_generator(0, device=input.device))
        r2 = generate_random_p_element([1], device=r.device, generator=comm.get().get_generator(0, device=input.device))

        fr = (r * r1.item() + r2.item()) % (1 << 31)

        req1 = comm.get().send(fr, src=rank, group=getattr(comm.get(), f"group{rank}{2}"))
        req1.wait()

        b = (cuda_tensor._tensor>>63)&1
         
    if rank==1:
        input=(abs(input)%((1<<31)-1)).int()
        input = input.reshape(1,-1).repeat(32,1)

        import torchcsprng as csprng
        urandom_gen=csprng.create_random_device_generator('/dev/urandom')
        r = torch.empty(input.shape,dtype=torch.int,device='cuda').random_(0, to=(2**31 - 1),generator=urandom_gen)
        tic = time.perf_counter()
        import encode
        encode.torch_launch_encode(input,r,input.shape[1])
        toc = time.perf_counter()
        # hash
        r1 = generate_random_p_element([1], device=r.device, generator=comm.get().get_generator(1, device=input.device))
        r2 = generate_random_p_element([1], device=r.device, generator=comm.get().get_generator(1, device=input.device))

        fr = (r * r1.item() + r2.item()) % (1 << 31)
        
        req1 = comm.get().send(fr, src=rank, group=getattr(comm.get(), f"group{rank}{2}"))
        req1.wait()

        b = 1 - (cuda_tensor._tensor>>63)&1
        
        
    if rank==2:
        
        x=torch.empty(cuda_tensor.reshape(1,-1).repeat(32,1).size(),dtype=torch.int,device='cuda')
        y=torch.empty(cuda_tensor.reshape(1,-1).repeat(32,1).size(),dtype=torch.int,device='cuda')
        
        req1 = comm.get().recv(x, src=0, group=comm.get().group02)
        req2 = comm.get().recv(y, src=1, group=comm.get().group12)
        req1.wait()
        req2.wait()

        z=x^y

        import eq
        eq.torch_launch_eq(z,z.shape[1])

        z=torch.sum(z,dim=0).view_as(cuda_tensor._tensor)

        b=(z%2)^1
        
    return BinarySharedTensor.from_shares(b, src=rank)



def new_get_msb(arithmetic_tensor):
    import crypten
    from crypten.mpc import MPCTensor
    from crypten.mpc.primitives import BinarySharedTensor,ArithmeticSharedTensor
    from crypten import communicator as comm
    #print(arithmetic_tensor.get_plain_text().flatten())
    input = arithmetic_tensor.share.clone()

    rank = comm.get().get_rank()

    rep_share = resharing.replicate_shares(input)
    
    if rank==0:
        input = (input + rep_share)
    
    
    return three_mill(input, rank)
    