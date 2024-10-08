#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm

# dependencies:
import torch
from crypten.common.rng import generate_kbit_random_tensor
from crypten.common.tensor_types import is_tensor
from crypten.common.util import torch_cat, torch_stack
from crypten.cuda import CUDALongTensor
from crypten.encoder import FixedPointEncoder

from . import circuit, resharing


SENTINEL = -1


# MPC tensor where shares are XOR-sharings.
class BinarySharedTensor(object):
    """
        Encrypted tensor object that uses binary sharing to perform computations.

        Binary shares are computed by splitting each value of the input tensor
        into n separate random values that xor together to the input tensor value,
        where n is the number of parties present in the protocol (world_size).
    """

    def __init__(self, tensor=None, size=None, src=0, device=None):
        self.rep_share = None

        if src == SENTINEL:
            return
        assert (
            isinstance(src, int) and src >= 0 and src < comm.get().get_world_size()
        ), "invalid tensor source"

        if device is None and hasattr(tensor, "device"):
            device = tensor.device

        #  Assume 0 bits of precision unless encoder is set outside of init
        self.encoder = FixedPointEncoder(precision_bits=0)
        if tensor is not None:
            tensor = self.encoder.encode(tensor)
            tensor = tensor.to(device=device)
            size = tensor.size()

        # Generate Psuedo-random Sharing of Zero and add source's tensor
        self.share = BinarySharedTensor.PRZS(size, device=device).share
        if self.rank == src:
            assert tensor is not None, "Source must provide a data tensor"
            if hasattr(tensor, "src"):
                assert (
                    tensor.src == src
                ), "Source of data tensor must match source of encryption"
            self.share ^= tensor

    @staticmethod
    def from_shares(share, precision=None, src=0, device=None):
        """Generate a BinarySharedTensor from a share from each party"""
        result = BinarySharedTensor(src=SENTINEL)
        share = share.to(device) if device is not None else share
        result.share = CUDALongTensor(share) if share.is_cuda else share
        result.encoder = FixedPointEncoder(precision_bits=precision)
        return result

    @staticmethod
    def PRZS(*size, device=None):
        """
        Generate a Pseudo-random Sharing of Zero (using arithmetic shares)

        This function does so by generating `n` numbers across `n` parties with
        each number being held by exactly 2 parties. Therefore, each party holds
        two numbers. A zero sharing is found by having each party xor their two
        numbers together.
        """
        tensor = BinarySharedTensor(src=SENTINEL)
        current_share = generate_kbit_random_tensor(
            *size, device=device, generator=comm.get().get_generator(0, device=device)
        )
        next_share = generate_kbit_random_tensor(
            *size, device=device, generator=comm.get().get_generator(1, device=device)
        )
        tensor.share = current_share ^ next_share
        return tensor

    @staticmethod
    def rand(*size, bits=64, device=None):
        """
        Generate a uniform random samples with a given size.
        """
        tensor = BinarySharedTensor(src=SENTINEL)
        if isinstance(size[0], (torch.Size, tuple)):
            size = size[0]
        tensor.share = generate_kbit_random_tensor(size, bitlength=bits, device=device)
        return tensor

    @property
    def device(self):
        """Return the `torch.device` of the underlying _tensor"""
        return self._tensor.device

    @property
    def is_cuda(self):
        """Return True if the underlying _tensor is stored on GPU, False otherwise"""
        return self._tensor.is_cuda

    def to(self, *args, **kwargs):
        """Call `torch.Tensor.to` on the underlying _tensor"""
        self._tensor = self._tensor.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Call `torch.Tensor.cuda` on the underlying _tensor"""
        self._tensor = CUDALongTensor(self._tensor.cuda(*args, **kwargs))
        return self

    def cpu(self, *args, **kwargs):
        """Call `torch.Tensor.cpu` on the underlying _tensor"""
        self._tensor = self._tensor.cpu(*args, **kwargs)
        return self

    @property
    def rank(self):
        return comm.get().get_rank()

    @property
    def share(self):
        """Returns underlying _tensor"""
        return self._tensor

    @share.setter
    def share(self, value):
        """Sets _tensor to value"""
        self._tensor = value

    def shallow_copy(self):
        """Create a shallow copy"""
        result = BinarySharedTensor(src=SENTINEL)
        result.encoder = self.encoder
        result.share = self.share
        result.rep_share = self.rep_share
        return result

    def copy_(self, other):
        """Copies other tensor into this tensor."""
        self.share.copy_(other.share)
        self.rep_share.copy_(other.rep_share)
        self.encoder = other.encoder

    def __repr__(self):
        return f"BinarySharedTensor({self.share})"

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate BinarySharedTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate BinarySharedTensors to boolean values")

    def __ixor__(self, y):
        """Bitwise XOR operator (element-wise) in place"""
        if is_tensor(y) or isinstance(y, int):
            if self.rank == 0:
                self.share ^= y
        elif isinstance(y, BinarySharedTensor):
            self.share ^= y.share
        else:
            raise TypeError("Cannot XOR %s with %s." % (type(y), type(self)))
        return self

    def __xor__(self, y):
        """Bitwise XOR operator (element-wise)"""
        result = self.clone()
        if isinstance(y, BinarySharedTensor):
            broadcast_tensors = torch.broadcast_tensors(result.share, y.share)
            result.share = broadcast_tensors[0].clone()
        elif is_tensor(y):
            broadcast_tensors = torch.broadcast_tensors(result.share, y)
            result.share = broadcast_tensors[0].clone()
        return result.__ixor__(y)

    def __iand__(self, y):
        """Bitwise AND operator (element-wise) in place"""
        if is_tensor(y) or isinstance(y, int):
            self.share &= y
        elif isinstance(y, BinarySharedTensor):
            assert comm.get().get_world_size() == 3
            self.share.set_(resharing.AND(self, y).share.data)
        else:
            raise TypeError("Cannot AND %s with %s." % (type(y), type(self)))
        return self

    def __and__(self, y):
        """Bitwise AND operator (element-wise)"""
        result = self.clone()
        if isinstance(y, BinarySharedTensor):
            broadcast_tensors = torch.broadcast_tensors(result.share, y.share)
            result.share = broadcast_tensors[0].clone()
        elif is_tensor(y):
            broadcast_tensors = torch.broadcast_tensors(result.share, y)
            result.share = broadcast_tensors[0].clone()
        return result.__iand__(y)

    def __ior__(self, y):
        """Bitwise OR operator (element-wise) in place"""
        xor_result = self ^ y
        return self.__iand__(y).__ixor__(xor_result)

    def __or__(self, y):
        """Bitwise OR operator (element-wise)"""
        return self.__and__(y) ^ self ^ y

    def __invert__(self):
        """Bitwise NOT operator (element-wise)"""
        result = self.clone()
        if result.rank == 0:
            result.share ^= -1
        return result

    def lshift_(self, value):
        """Left shift elements by `value` bits"""
        assert isinstance(value, int), "lshift must take an integer argument."
        self.share <<= value
        return self

    def lshift(self, value):
        """Left shift elements by `value` bits"""
        return self.clone().lshift_(value)

    def rshift_(self, value):
        """Right shift elements by `value` bits"""
        assert isinstance(value, int), "rshift must take an integer argument."
        self.share >>= value
        return self

    def rshift(self, value):
        """Right shift elements by `value` bits"""
        return self.clone().rshift_(value)

    # Circuits
    # def add(self, y):
    #     """Compute [self] + [y] for xor-sharing"""
    #     return circuit.add(self, y)

    # def eq(self, y):
    #     return circuit.eq(self, y)

    # def ne(self, y):
    #     return self.eq(y) ^ 1

    # def lt(self, y):
    #     return circuit.lt(self, y)

    # def le(self, y):
    #     return circuit.le(self, y)

    # def gt(self, y):
    #     return circuit.gt(self, y)

    # def ge(self, y):
    #     return circuit.ge(self, y)

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if is_tensor(value) or isinstance(value, list):
            value = BinarySharedTensor(value)
        assert isinstance(
            value, BinarySharedTensor
        ), "Unsupported input type %s for __setitem__" % type(value)
        self.share.__setitem__(index, value.share)

    @staticmethod
    def stack(seq, *args, **kwargs):
        """Stacks a list of tensors along a given dimension"""
        assert isinstance(seq, list), "Stack input must be a list"
        assert isinstance(
            seq[0], BinarySharedTensor
        ), "Sequence must contain BinarySharedTensors"
        result = seq[0].shallow_copy()
        result.share = torch_stack(
            [BinarySharedTensor.share for BinarySharedTensor in seq], *args, **kwargs
        )
        return result

    @staticmethod
    def reveal_batch(tensor_or_list, dst=None):
        """Get (batched) plaintext without any downscaling"""
        if isinstance(tensor_or_list, BinarySharedTensor):
            return tensor_or_list.reveal(dst=dst)

        assert isinstance(
            tensor_or_list, list
        ), f"Invalid input type into reveal {type(tensor_or_list)}"
        shares = [tensor.share for tensor in tensor_or_list]
        op = torch.distributed.ReduceOp.BXOR
        if dst is None:
            return comm.get().all_reduce(shares, op=op, batched=True)
        else:
            return comm.get().reduce(shares, dst=dst, op=op, batched=True)

    def reveal(self, dst=None):
        """Get plaintext without any downscaling"""
        op = torch.distributed.ReduceOp.BXOR
        if dst is None:
            return comm.get().all_reduce(self.share, op=op)
        else:
            return comm.get().reduce(self.share, dst=dst, op=op)

    def get_plain_text(self, dst=None):
        """Decrypts the tensor."""
        # Edge case where share becomes 0 sized (e.g. result of split)
        if self.nelement() < 1:
            return torch.empty(self.share.size())
        #return self.encoder.decode(self.reveal(dst=dst))
        return self.reveal(dst=dst)

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or BinarySharedTensor): when True yield self,
                otherwise yield y. Note condition is not bitwise.
            y (torch.tensor or BinarySharedTensor): selected when condition is
                False.

        Returns: BinarySharedTensor or torch.tensor.
        """
        if is_tensor(condition):
            condition = condition.long()
            is_binary = ((condition == 1) | (condition == 0)).all()
            assert is_binary, "condition values must be 0 or 1"
            # -1 mult expands 0 into binary 00...00 and 1 into 11...11
            condition_expanded = -condition
            y_masked = y & (~condition_expanded)
        elif isinstance(condition, BinarySharedTensor):
            condition_expanded = condition.clone()
            # -1 mult expands binary while & 1 isolates first bit
            condition_expanded.share = -(condition_expanded.share & 1)
            # encrypted tensor must be first operand
            y_masked = (~condition_expanded) & y
        else:
            msg = f"condition {condition} must be torch.bool, or BinarySharedTensor"
            raise ValueError(msg)

        return (self & condition_expanded) ^ y_masked

    def scatter_(self, dim, index, src):
        """Writes all values from the tensor `src` into `self` at the indices
        specified in the `index` tensor. For each value in `src`, its output index
        is specified by its index in `src` for `dimension != dim` and by the
        corresponding value in `index` for `dimension = dim`.
        """
        if is_tensor(src):
            src = BinarySharedTensor(src)
        assert isinstance(
            src, BinarySharedTensor
        ), "Unrecognized scatter src type: %s" % type(src)
        self.share.scatter_(dim, index, src.share)
        return self

    def scatter(self, dim, index, src):
        """Writes all values from the tensor `src` into `self` at the indices
        specified in the `index` tensor. For each value in `src`, its output index
        is specified by its index in `src` for `dimension != dim` and by the
        corresponding value in `index` for `dimension = dim`.
        """
        result = self.clone()
        return result.scatter_(dim, index, src)

    def prod(self, dim=None, keepdim=False):
        """
        Returns the product of each row of the `input` tensor in the given
        dimension `dim`.

        If `keepdim` is `True`, the output tensor is of the same size as `input`
        except in the dimension `dim` where it is of size 1. Otherwise, `dim` is
        squeezed, resulting in the output tensor having 1 fewer dimension than
        `input`.
        """
        if dim is None:
            return self.flatten().prod(dim=0)

        result = self.clone()
        while result.size(dim) > 1:
            size = result.size(dim)
            x, y, remainder = result.split([size // 2, size // 2, size % 2], dim=dim)
            result = x & y

            result.share = torch_cat([result.share, remainder.share], dim=dim)

        # Squeeze result if necessary
        if not keepdim:
            result.share = result.share.squeeze(dim)
        return result

    # Bitwise operators
    # __add__ = add
    # __eq__ = eq
    # __ne__ = ne
    # __lt__ = lt
    # __le__ = le
    # __gt__ = gt
    # __ge__ = ge
    __lshift__ = lshift
    __rshift__ = rshift

    # In-place bitwise operators
    __ilshift__ = lshift_
    __irshift__ = rshift_

    # Reversed boolean operations
    # __radd__ = __add__
    __rxor__ = __xor__
    __rand__ = __and__
    __ror__ = __or__


REGULAR_FUNCTIONS = [
    "clone",
    "__getitem__",
    "index_select",
    "view",
    "flatten",
    "t",
    "transpose",
    "unsqueeze",
    "squeeze",
    "repeat",
    "narrow",
    "expand",
    "roll",
    "unfold",
    "flip",
    "reshape",
    "gather",
    "take",
    "split",
    "permute",
]


PROPERTY_FUNCTIONS = ["__len__", "nelement", "dim", "size", "numel"]


def _add_regular_function(function_name):
    def regular_func(self, *args, **kwargs):
        result = self.shallow_copy()
        result.share = getattr(result.share, function_name)(*args, **kwargs)
        return result

    setattr(BinarySharedTensor, function_name, regular_func)


def _add_property_function(function_name):
    def property_func(self, *args, **kwargs):
        return getattr(self.share, function_name)(*args, **kwargs)

    setattr(BinarySharedTensor, function_name, property_func)


for function_name in REGULAR_FUNCTIONS:
    _add_regular_function(function_name)

for function_name in PROPERTY_FUNCTIONS:
    _add_property_function(function_name)
