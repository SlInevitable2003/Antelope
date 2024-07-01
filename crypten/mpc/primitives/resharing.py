#!/usr/bin/env python3
import crypten
import crypten.communicator as comm
import torch

from crypten.cuda import CUDALongTensor
from crypten.common.util import torch_stack
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.encoder import FixedPointEncoder
from crypten.common.util import im2col_indices, col2im_indices

import time

def replicate_shares(x_share):
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    x_rep = torch.zeros_like(x_share.data)

    send_group = getattr(comm.get(), f"group{rank}{next_rank}")
    recv_group = getattr(comm.get(), f"group{prev_rank}{rank}")

    tic = time.perf_counter()


    req1 = comm.get().isend(x_share.data, dst=next_rank, group=send_group)
    req2 = comm.get().irecv(x_rep.data, src=prev_rank, group=recv_group)

    req1.wait()
    req2.wait()

    toc = time.perf_counter()
    
    comm.get().comm_time += toc - tic

    if x_share.is_cuda:
        x_rep = CUDALongTensor(x_rep)

    return x_rep

def _special_matmul(x,y):
    x1, y1 = x.share, y.share
    m = x1.shape[0]
    n = x1.shape[1]
    k = y1.shape[1]
    # first matmul x,y
    z_1 = torch.empty([m,k],dtype=torch.int64,device='cuda')
    x1 = x1._tensor.contiguous()
    y1 = y1._tensor.contiguous()
    import mat_mul
    mat_mul.torch_launch_matmul(z_1, x1, y1, m, n, k)

    # communication
    tic = time.perf_counter()
    x2 = replicate_shares(x.share) if x.rep_share is None else x.rep_share
    toc = time.perf_counter()
    #print("com",(toc-tic))
    x2 = x2._tensor.contiguous()
    z_2 = torch.empty([m,k],dtype=torch.int64,device='cuda')
    mat_mul.torch_launch_matmul(z_2, x2, y1, m, n, k)


    y2 = replicate_shares(y.share) if y.rep_share is None else y.rep_share
    y2 = y2.view_as(y1)
    # second matmul other
    y2 = y2._tensor.contiguous()
    z_3 = torch.empty([m,k],dtype=torch.int64,device='cuda')
    mat_mul.torch_launch_matmul(z_3, x1, y2, m, n, k)

    return CUDALongTensor(z_1+z_2+z_3)
def _crypt_matmul(x,y):
    x1, y1 = x.share, y.share
    x2 = replicate_shares(x.share) if x.rep_share is None else x.rep_share
    y2 = replicate_shares(y.share) if y.rep_share is None else y.rep_share

    y2 = y2.view_as(y1)
    
    z = getattr(torch, "matmul")(x1, y1) + getattr(torch, "matmul")(x1, y2) + getattr(torch, "matmul")(x2, y1)
    return z
def _conv(x,y,*args, **kwargs):
    x1, y1 = x.share, y.share
    x2 = replicate_shares(x.share) if x.rep_share is None else x.rep_share
    y2 = replicate_shares(y.share) if y.rep_share is None else y.rep_share

    y2 = y2.view_as(y1)
    
    z = CUDALongTensor.conv2d(x1, y1,*args, **kwargs) + CUDALongTensor.conv2d(x1, y2,*args, **kwargs) + CUDALongTensor.conv2d(x2, y1,*args, **kwargs)
    return z
def _conv2d_back(x, y, *args, **kwargs):
    #x: input y: grad_out
    kernel = kwargs["kernel"]
    out_channels, in_channels, kernel_size_y, kernel_size_x = kernel.size()
    padding = kwargs["padding"]
    stride = kwargs["stride"]

    input_fold = im2col_indices(x, kernel_size_y, kernel_size_x, padding=padding, stride=stride)
    grad_kernel = getattr(torch, "matmul")(y, input_fold.transpose(0,1)._tensor._tensor)
    return grad_kernel.view(kernel.shape)


    print(kernel.shape)
def __replicated_secret_sharing_protocol(op, x, y, *args, **kwargs):
    assert op in {
        "mul",
        "matmul",
        "conv1d",
        "conv2d",
        "conv2d_back",
        "conv_transpose1d",
        "conv_transpose2d",
    }
    from .arithmetic import ArithmeticSharedTensor
    from .binary import BinarySharedTensor
    if op =="conv2d_back":
        x1, y1 = x.share, y.share
        x2 = replicate_shares(x.share) if x.rep_share is None else x.rep_share
        y2 = replicate_shares(y.share) if y.rep_share is None else y.rep_share

        y2 = y2.view_as(y1)
        
        z = _conv2d_back(x1, y1, *args, **kwargs) + _conv2d_back(x1, y2, *args, **kwargs) + _conv2d_back(x2, y1, *args, **kwargs)

    elif op=="matmul" and x.dim()==2:
        z = _special_matmul(x,y)
    else:
        x1, y1 = x.share, y.share
        x2 = replicate_shares(x.share) if x.rep_share is None else x.rep_share
        y2 = replicate_shares(y.share) if y.rep_share is None else y.rep_share

        y2 = y2.view_as(y1)
        
        z = getattr(torch, op)(x1, y1, *args, **kwargs) + getattr(torch, op)(x1, y2, *args, **kwargs) + getattr(torch, op)(x2, y1, *args, **kwargs)

    rank = comm.get().get_rank()
    
    if isinstance(x, BinarySharedTensor):
        z = BinarySharedTensor.from_shares(z, src=rank)
        z += BinarySharedTensor.PRZS(z.size(), device=z.device)
    elif isinstance(x, ArithmeticSharedTensor):
        z = ArithmeticSharedTensor.from_shares(z, src=rank)
        z += ArithmeticSharedTensor.PRZS(z.size(), device=z.device)

    return z


def mul(x, y):
    return __replicated_secret_sharing_protocol("mul", x, y)


def matmul(x, y):
    return __replicated_secret_sharing_protocol("matmul", x, y)


def conv1d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv1d", x, y, **kwargs)


def conv2d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv2d", x, y, **kwargs)
def conv2d_back(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv2d_back", x, y, **kwargs)

def conv_transpose1d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv_transpose1d", x, y, **kwargs)


def conv_transpose2d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv_transpose2d", x, y, **kwargs)


def square(x):
    from .arithmetic import ArithmeticSharedTensor

    x1 = x.share
    x2 = replicate_shares(x.share)
    x_square = x1 ** 2 + 2 * x1 * x2

    return ArithmeticSharedTensor.from_shares(x_square, src=comm.get().get_rank())


def truncation(x, y):
    rank = x.rank
    scale = y
    #scale = x.encoder.scale

    rep_share = replicate_shares(x.share)
    
    r = None
    # Step 1: Parties 2,3 locally compute a random r in Z_2^k .
    if rank == 1 or rank == 0:
        if rank == 1:
            r = generate_random_ring_element(x.share.size(), device=x.device, generator=comm.get().get_generator(1, device=x.device))
        if rank == 0:
            r = generate_random_ring_element(x.share.size(), device=x.device, generator=comm.get().get_generator(0, device=x.device))
        
    # Step 2: Party 1,3 locally compute x1 := x/2^d
    if rank == 0 or rank == 2:
        if rank == 2:
            x.share //= scale
        if rank == 0:
            rep_share //= scale
    # Party 2 locally computes x2 := (x
    if rank == 1:
        x.share = (x.share + rep_share) // scale - r

    if rank == 1 or rank == 0:
        if rank == 0:
            x.share = r

    return x


def pack_bits(tensor):
    tensor = tensor.view(-1)
    tensor = tensor & 1
    num_param = tensor.size(0)
    pad = (8 - num_param % 8) % 8
    num_groups = num_param // 8 + (pad > 0) * 1
    
    tensor_pad = torch.cat([tensor.data, torch.zeros([pad], device=tensor.device).long()])
    tensor_pad = tensor_pad.view(8,-1)

    tensor_pack = torch.zeros(num_groups, dtype=torch.uint8, device=tensor.device)
    
    for i in range(8):
        tensor_pack ^= (tensor_pad[i] << i)

    return tensor_pack
        

def unpack_bits(tensor_pack, size):
    import numpy as np
    tensor_pad = torch_stack([
        (tensor_pack >> (i)) & 1  for i in range(8)
    ])    

    tensor_pad = tensor_pad.view(-1)
    num_param =  int(np.prod(list(size)))

    tensor = tensor_pad[:num_param]
    
    return tensor.view(size).long()


def replicate_shares_bit(x, bit=1):
    from .binary import BinarySharedTensor

    x_share = x
    if isinstance(x, BinarySharedTensor):
        x_share = x.share

    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    prev_rank = (rank-1) % world_size
    next_rank = (rank+1) % world_size

    size = x.size()

    share_bit = (x_share >> (bit-1)) & 1
    share_bit = pack_bits(share_bit)

    share_bit = share_bit.contiguous()

    rep_share = torch.zeros_like(share_bit).byte()

    send_group = getattr(comm.get(), f"group{rank}{next_rank}")
    recv_group = getattr(comm.get(), f"group{prev_rank}{rank}")

    tic = time.perf_counter()
    req0 = comm.get().isend(share_bit, dst=next_rank, group=send_group)
    req1 = comm.get().irecv(rep_share, src=prev_rank, group=recv_group)

    req0.wait()
    req1.wait()

    toc = time.perf_counter()

    comm.get().comm_time += toc - tic

    rep_share = unpack_bits(rep_share, size)

    if isinstance(x, BinarySharedTensor):
        res = BinarySharedTensor.from_shares(rep_share, src=rank)
        res.encoder = x.encoder
        return res

    return rep_share


def AND(x, y):
    from .binary import BinarySharedTensor
    x1, x2, y1, y2 = None, None, None, None

    x1, x2 = x1, replicate_shares(x)
    
    if isinstance(y, BinarySharedTensor):
        y1 = y.share
        y2 = replicate_shares(y1) if y.rep_share is None else y.rep_share
    else:
        y1, y2 = y, replicate_shares(y)

    return BinarySharedTensor.from_shares((x1 & y1) ^ (x2 & y1) ^ (x1 & y2), src=comm.get().get_rank())


def AND_gate(xB, yB):
    from .binary import BinarySharedTensor

    if isinstance(xB, BinarySharedTensor):
        xB = xB.share
    if isinstance(yB, BinarySharedTensor):
        yB = yB.share

    x1, x2 = xB.data, replicate_shares_bit(xB, bit=1)
    y1, y2 = yB.data, replicate_shares_bit(yB, bit=1)

    x1, y1 = x1 & 1, y1 & 1

    res = (x1 & y1) ^ (x2 & y1) ^ (x1 & y2)
    res = BinarySharedTensor.from_shares(res, src=comm.get().get_rank())
    return res

def new_mixed_mul(x,yB,bits=None):
    from .arithmetic import ArithmeticSharedTensor
    from .binary import BinarySharedTensor
    from crypten.mpc import MPCTensor
   
    if bits is None:
        bits = torch.iinfo(torch.long).bits

    zero_share = ArithmeticSharedTensor.PRZS(yB.size(), device=yB.device).share
    
    rank = yB.rank
    y = yB.share.clone()

    size = y.size()
    
    if rank==2:
        import torchcsprng as csprng
        urandom_gen=csprng.create_random_device_generator('/dev/urandom')
        r1 =torch.empty(size,dtype=torch.int64,device='cuda').random_(-(2**64 // 2), to=(2**64 - 1) // 2,generator=urandom_gen)
        
        r2 =y-r1

        req1 = comm.get().send(r1, src=2, group=comm.get().group02)
        req2 = comm.get().send(r2, src=2, group=comm.get().group12)
        req1.wait()
        req2.wait()

        
        a = ArithmeticSharedTensor.from_shares(zero_share, src=rank)
        
    if rank==1:
        r =torch.empty(size,dtype=torch.int64,device='cuda')
        req1 = comm.get().recv(r, src=2, group=comm.get().group12)
        req1.wait()
        z = y._tensor
        import c_sub
        c_sub.torch_launch_csub(z, r, y.numel(), rank)
        a = ArithmeticSharedTensor.from_shares(zero_share + z, src=rank)
        
    if rank==0:
        r =torch.empty(size,dtype=torch.int64,device='cuda')
        req1 = comm.get().recv(r, src=2, group=comm.get().group02)
        req1.wait()
        z = y._tensor
        
        import c_sub
        c_sub.torch_launch_csub(z, r, y.numel(), rank)
        a = ArithmeticSharedTensor.from_shares(zero_share + z, src=rank)
    
    a.encoder = x.encoder

    a.encoder = FixedPointEncoder(precision_bits=None)
    scale = a.encoder._scale
    
    x1, a1 = x.share, a.share
    x2 = replicate_shares(x.share) if x.rep_share is None else x.rep_share
    a2 = replicate_shares(a.share) if a.rep_share is None else a.rep_share

    a2 = a2.view_as(a1)
    
    z = getattr(torch, 'mul')(x1, a1) + getattr(torch, 'mul')(x1, a2) + getattr(torch, 'mul')(x2, a1)
    
    return MPCTensor.from_shares(z + zero_share)
    
    
# Performing a three party OT
def mixed_mul(x, yB, bits=None):
    from .arithmetic import ArithmeticSharedTensor
    from .binary import BinarySharedTensor
    from crypten.mpc import MPCTensor
   
    if bits is None:
        bits = torch.iinfo(torch.long).bits
    
    rank = yB.rank
    if rank == 0:
        a = mixed_mul_scalar(torch.ones_like(yB.share.data), yB, bits=1, roles=[0,1,2])
    if rank == 1:
        a = mixed_mul_scalar(None, yB, bits=1, roles=[0,1,2])
    if rank == 2:
        a = mixed_mul_scalar(None, yB, bits=1, roles=[0,1,2])

    #print(x)
    a.encoder = x.encoder

    a.encoder = FixedPointEncoder(precision_bits=None)
    scale = a.encoder._scale
    a *= scale
    
    return MPCTensor.from_shares((x * a).share)


# Performing a three party OT
def mixed_mul_scalar(xs, yB, bits=1, roles=[0,1,2]):
    from .arithmetic import ArithmeticSharedTensor
    from .binary import BinarySharedTensor
    from crypten.mpc import MPCTensor
    # mixed multiply. x is a scalar, y is a BinarySharedTensor 
    # output their product as an ArithmeticSharedTensor
    assert isinstance(yB, BinarySharedTensor)
    
    sender, receiver, helper = roles
    groupsr = getattr(comm.get(), f"group{sender}{receiver}")
    groupsh = getattr(comm.get(), f"group{sender}{helper}")
    grouprh = getattr(comm.get(), f"group{receiver}{helper}")

    zero_share = ArithmeticSharedTensor.PRZS(yB.size(), device=yB.device).share
    #print(zero_share)
    if bits == None:
        bits = torch.iinfo(torch.long).bits

    rank = yB.rank
    if rank == sender:
        assert xs is not None
        assert xs.size() == yB.size()
        b1, b3 = yB.share, replicate_shares(yB.share)

        b1, b3 = b1 & 1, b3 & 1

        if sender > helper:
            b1, b3 = b3, b1

        r = generate_kbit_random_tensor(size=yB.size(), bitlength=64, device=yB.device)

        m0 = (b1 ^ b3 ^ 0) * xs - r
        m1 = (b1 ^ b3 ^ 1) * xs - r

        w0 = generate_kbit_random_tensor(size=yB.size(), bitlength=64, device=yB.device)
        w1 = generate_kbit_random_tensor(size=yB.size(), bitlength=64, device=yB.device)

        req0 = comm.get().isend(torch_stack([m0 ^ w0, m1 ^ w1]), dst=receiver, group=groupsr)
        req1 = comm.get().isend(torch_stack([w0, w1]), dst=helper, group=groupsh)
        req0.wait()
        req1.wait()
        
        return ArithmeticSharedTensor.from_shares(zero_share+r, src=rank) 

    if rank == receiver:
        b2, b1 = yB.share, replicate_shares(yB.share)

        if sender > helper:
            b1, b2 = b2, b1
    
        m_b = torch.zeros_like(torch_stack([b1,b2])).data
        w_b2 = torch.zeros_like(b2).data
        
        req0 = comm.get().irecv(m_b, src=sender, group=groupsr)
        req1 = comm.get().irecv(w_b2, src=helper, group=grouprh)
        req0.wait()
        req1.wait()

        size = b1.size()
        bin_bits = b2.flatten().data
        m_b = m_b.view(2, -1)

        m_b2 = m_b[bin_bits, torch.arange(bin_bits.size(0))]
        m_b2 = m_b2.view(size)

        message = m_b2 ^ w_b2

        #print(message)
        return ArithmeticSharedTensor.from_shares(zero_share+message, src=rank)

    if rank == helper:
        b3, b2 = yB.share, replicate_shares(yB.share)

        if sender > helper:
            b3, b2 = b2, b3

        w = torch.zeros_like(torch_stack([b3,b3])).data
        req0 = comm.get().irecv(w, src=sender, group=groupsh)
        req0.wait()

        size = b3.size()
        bin_bits = b2.flatten().data
        
        w = w.view(2, -1)
        w_b2 = w[bin_bits, torch.arange(bin_bits.size(0))]
        w_b2 = w_b2.view(size)

        req1 = comm.get().isend(w_b2, dst=receiver, group=grouprh)
        req1.wait()
        #print(rank)
        return ArithmeticSharedTensor.from_shares(zero_share, src=rank)
