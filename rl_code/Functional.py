r"""Functional interface"""
from typing import Callable, List, Optional, Tuple
import math
import warnings

import torch
from torch import _VF
from torch._C import _infer_size, _add_docstr#, _has_torch_function_variadic, _has_torch_function_unary
from torch._torch_docs import tf32_notes #reproducibility_notes,

from torch.overrides import (
    has_torch_function, #has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)



Tensor = torch.Tensor

reproducibility_notes = {
    "forward_reproducibility_note": """This operation may behave nondeterministically when given tensors on \
a CUDA device. See :doc:`/notes/randomness` for more information.""",
    "backward_reproducibility_note": """This operation may produce nondeterministic gradients when given tensors on \
a CUDA device. See :doc:`/notes/randomness` for more information.""",
    "cudnn_reproducibility_note": """In some circumstances when given tensors on a CUDA device \
and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is \
undesirable, you can try to make the operation deterministic (potentially at \
a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. \
See :doc:`/notes/randomness` for more information."""
}
# has_torch_function_unary = _add_docstr(
#     _has_torch_function_unary,
#     r"""Special case of `has_torch_function` for single inputs.
#     Instead of:
#       `has_torch_function((t,))`
#     call:
#       `has_torch_function_unary(t)`
#     which skips unnecessary packing and unpacking work.
#     """
# )
# has_torch_function_variadic = _add_docstr(
#     _has_torch_function_variadic,
#     r"""Special case of `has_torch_function` that skips tuple creation.
#
#     This uses the METH_FASTCALL protocol introduced in Python 3.7; for 3.6
#     and before it has roughly equivilent performance compared to
#     `has_torch_function`.
#
#     Instead of:
#       `has_torch_function((a, b))`
#     call:
#       `has_torch_function_variadic(a, b)`
#     which skips unnecessary packing and unpacking work.
#     """
# )

# conv1d = _add_docstr(
#     torch.conv1d,
#     r"""
# conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
# Applies a 1D convolution over an input signal composed of several input
# planes.
# {tf32_note}
# See :class:`~torch.nn.Conv1d` for details and output shape.
# Note:
#     {cudnn_reproducibility_note}
# """.format(
#         **reproducibility_notes, **tf32_notes
#     )
#     + r"""
# Args:
#     input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
#     weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)`
#     bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
#     stride: the stride of the convolving kernel. Can be a single number or
#       a one-element tuple `(sW,)`. Default: 1
#     padding: implicit paddings on both sides of the input. Can be a
#       single number or a one-element tuple `(padW,)`. Default: 0
#     dilation: the spacing between kernel elements. Can be a single number or
#       a one-element tuple `(dW,)`. Default: 1
#     groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
#       the number of groups. Default: 1
# Examples::
#     >>> filters = torch.randn(33, 16, 3)
#     >>> inputs = torch.randn(20, 16, 50)
#     >>> F.conv1d(inputs, filters)
# """,
# )

# conv2d = _add_docstr(
#     torch.conv2d,
#     r"""
# conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
# Applies a 2D convolution over an input image composed of several input
# planes.
# {tf32_note}
# See :class:`~torch.nn.Conv2d` for details and output shape.
# Note:
#     {cudnn_reproducibility_note}
# """.format(
#         **reproducibility_notes, **tf32_notes
#     )
#     + r"""
# Args:
#     input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
#     weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
#     bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
#     stride: the stride of the convolving kernel. Can be a single number or a
#       tuple `(sH, sW)`. Default: 1
#     padding: implicit paddings on both sides of the input. Can be a
#       single number or a tuple `(padH, padW)`. Default: 0
#     dilation: the spacing between kernel elements. Can be a single number or
#       a tuple `(dH, dW)`. Default: 1
#     groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
#       number of groups. Default: 1
# Examples::
#     >>> # With square kernels and equal stride
#     >>> filters = torch.randn(8,4,3,3)
#     >>> inputs = torch.randn(1,4,5,5)
#     >>> F.conv2d(inputs, filters, padding=1)
# """,
# )  # noqa: E501

# conv3d = _add_docstr(
#     torch.conv3d,
#     r"""
# conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
# Applies a 3D convolution over an input image composed of several input
# planes.
# {tf32_note}
# See :class:`~torch.nn.Conv3d` for details and output shape.
# Note:
#     {cudnn_reproducibility_note}
# """.format(
#         **reproducibility_notes, **tf32_notes
#     )
#     + r"""
# Args:
#     input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
#     weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)`
#     bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: None
#     stride: the stride of the convolving kernel. Can be a single number or a
#       tuple `(sT, sH, sW)`. Default: 1
#     padding: implicit paddings on both sides of the input. Can be a
#       single number or a tuple `(padT, padH, padW)`. Default: 0
#     dilation: the spacing between kernel elements. Can be a single number or
#       a tuple `(dT, dH, dW)`. Default: 1
#     groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
#       the number of groups. Default: 1
# Examples::
#     >>> filters = torch.randn(33, 16, 3, 3, 3)
#     >>> inputs = torch.randn(20, 16, 50, 10, 20)
#     >>> F.conv3d(inputs, filters)
# """,
# )  # noqa: E501

# conv_transpose1d = _add_docstr(
#     torch.conv_transpose1d,
#     r"""
# conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor
# Applies a 1D transposed convolution operator over an input signal
# composed of several input planes, sometimes also called "deconvolution".
# {tf32_note}
# See :class:`~torch.nn.ConvTranspose1d` for details and output shape.
# Note:
#     {cudnn_reproducibility_note}
# """.format(
#         **reproducibility_notes, **tf32_notes
#     )
#     + r"""
# Args:
#     input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
#     weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kW)`
#     bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
#     stride: the stride of the convolving kernel. Can be a single number or a
#       tuple ``(sW,)``. Default: 1
#     padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
#       sides of each dimension in the input. Can be a single number or a tuple
#       ``(padW,)``. Default: 0
#     output_padding: additional size added to one side of each dimension in the
#       output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0
#     groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
#       number of groups. Default: 1
#     dilation: the spacing between kernel elements. Can be a single number or
#       a tuple ``(dW,)``. Default: 1
# Examples::
#     >>> inputs = torch.randn(20, 16, 50)
#     >>> weights = torch.randn(16, 33, 5)
#     >>> F.conv_transpose1d(inputs, weights)
# """,
# )

# conv_transpose2d = _add_docstr(
#     torch.conv_transpose2d,
#     r"""
# conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor
# Applies a 2D transposed convolution operator over an input image
# composed of several input planes, sometimes also called "deconvolution".
# {tf32_note}
# See :class:`~torch.nn.ConvTranspose2d` for details and output shape.
# Note:
#     {cudnn_reproducibility_note}
# """.format(
#         **reproducibility_notes, **tf32_notes
#     )
#     + r"""
# Args:
#     input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
#     weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kH , kW)`
#     bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
#     stride: the stride of the convolving kernel. Can be a single number or a
#       tuple ``(sH, sW)``. Default: 1
#     padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
#       sides of each dimension in the input. Can be a single number or a tuple
#       ``(padH, padW)``. Default: 0
#     output_padding: additional size added to one side of each dimension in the
#       output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.
#       Default: 0
#     groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
#       number of groups. Default: 1
#     dilation: the spacing between kernel elements. Can be a single number or
#       a tuple ``(dH, dW)``. Default: 1
# Examples::
#     >>> # With square kernels and equal stride
#     >>> inputs = torch.randn(1, 4, 5, 5)
#     >>> weights = torch.randn(4, 8, 3, 3)
#     >>> F.conv_transpose2d(inputs, weights, padding=1)
# """,
# )  # noqa: E501

# conv_transpose3d = _add_docstr(
#     torch.conv_transpose3d,
#     r"""
# conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor
# Applies a 3D transposed convolution operator over an input image
# composed of several input planes, sometimes also called "deconvolution"
# {tf32_note}
# See :class:`~torch.nn.ConvTranspose3d` for details and output shape.
# Note:
#     {cudnn_reproducibility_note}
# """.format(
#         **reproducibility_notes, **tf32_notes
#     )
#     + r"""
# Args:
#     input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
#     weight: filters of shape :math:`(\text{in\_channels} , \frac{\text{out\_channels}}{\text{groups}} , kT , kH , kW)`
#     bias: optional bias of shape :math:`(\text{out\_channels})`. Default: None
#     stride: the stride of the convolving kernel. Can be a single number or a
#       tuple ``(sT, sH, sW)``. Default: 1
#     padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
#       sides of each dimension in the input. Can be a single number or a tuple
#       ``(padT, padH, padW)``. Default: 0
#     output_padding: additional size added to one side of each dimension in the
#       output shape. Can be a single number or a tuple
#       ``(out_padT, out_padH, out_padW)``. Default: 0
#     groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
#       number of groups. Default: 1
#     dilation: the spacing between kernel elements. Can be a single number or
#       a tuple `(dT, dH, dW)`. Default: 1
# Examples::
#     >>> inputs = torch.randn(20, 16, 50, 10, 20)
#     >>> weights = torch.randn(16, 33, 3, 3, 3)
#     >>> F.conv_transpose3d(inputs, weights)
# """,
# )  # noqa: E501

# conv_tbc = _add_docstr(
#     torch.conv_tbc,
#     r"""
# Applies a 1-dimensional sequence convolution over an input sequence.
# Input and output dimensions are (Time, Batch, Channels) - hence TBC.
# Args:
#     input: input tensor of shape :math:`(\text{sequence length} \times batch \times \text{in\_channels})`
#     weight: filter of shape (:math:`\text{kernel width} \times \text{in\_channels} \times \text{out\_channels}`)
#     bias: bias of shape (:math:`\text{out\_channels}`)
#     pad: number of timesteps to pad. Default: 0
# """,
# )


# Pooling
# avg_pool1d = _add_docstr(
#     torch.avg_pool1d,
#     r"""
# avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor
# Applies a 1D average pooling over an input signal composed of several
# input planes.
# See :class:`~torch.nn.AvgPool1d` for details and output shape.
# Args:
#     input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
#     kernel_size: the size of the window. Can be a single number or a
#       tuple `(kW,)`
#     stride: the stride of the window. Can be a single number or a tuple
#       `(sW,)`. Default: :attr:`kernel_size`
#     padding: implicit zero paddings on both sides of the input. Can be a
#       single number or a tuple `(padW,)`. Default: 0
#     ceil_mode: when True, will use `ceil` instead of `floor` to compute the
#         output shape. Default: ``False``
#     count_include_pad: when True, will include the zero-padding in the
#         averaging calculation. Default: ``True``
# Examples::
#     >>> # pool of square window of size=3, stride=2
#     >>> input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)
#     >>> F.avg_pool1d(input, kernel_size=3, stride=2)
#     tensor([[[ 2.,  4.,  6.]]])
# """,
# )


# avg_pool2d = _add_docstr(
#     torch._C._nn.avg_pool2d,
#     r"""
# avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor
# Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size
# :math:`sH \times sW` steps. The number of output features is equal to the number of
# input planes.
# See :class:`~torch.nn.AvgPool2d` for details and output shape.
# Args:
#     input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
#     kernel_size: size of the pooling region. Can be a single number or a
#       tuple `(kH, kW)`
#     stride: stride of the pooling operation. Can be a single number or a
#       tuple `(sH, sW)`. Default: :attr:`kernel_size`
#     padding: implicit zero paddings on both sides of the input. Can be a
#       single number or a tuple `(padH, padW)`. Default: 0
#     ceil_mode: when True, will use `ceil` instead of `floor` in the formula
#         to compute the output shape. Default: ``False``
#     count_include_pad: when True, will include the zero-padding in the
#         averaging calculation. Default: ``True``
#     divisor_override: if specified, it will be used as divisor, otherwise
#          size of the pooling region will be used. Default: None
# """,
# )

# avg_pool3d = _add_docstr(
#     torch._C._nn.avg_pool3d,
#     r"""
# avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor
# Applies 3D average-pooling operation in :math:`kT \times kH \times kW` regions by step
# size :math:`sT \times sH \times sW` steps. The number of output features is equal to
# :math:`\lfloor\frac{\text{input planes}}{sT}\rfloor`.
# See :class:`~torch.nn.AvgPool3d` for details and output shape.
# Args:
#     input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iT \times iH , iW)`
#     kernel_size: size of the pooling region. Can be a single number or a
#       tuple `(kT, kH, kW)`
#     stride: stride of the pooling operation. Can be a single number or a
#       tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
#     padding: implicit zero paddings on both sides of the input. Can be a
#       single number or a tuple `(padT, padH, padW)`, Default: 0
#     ceil_mode: when True, will use `ceil` instead of `floor` in the formula
#         to compute the output shape
#     count_include_pad: when True, will include the zero-padding in the
#         averaging calculation
#     divisor_override: if specified, it will be used as divisor, otherwise
#         size of the pooling region will be used. Default: None
# """,
# )

# Activation functions
def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.
    See :class:`~torch.nn.Dropout` for details.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    # if has_torch_function_unary(input):
    #     return handle_torch_function(dropout, (input,), input, p=p, training=training, inplace=inplace)
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)

def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    warnings.warn(
        "Implicit dimension choice for {} has been deprecated. "
        "Change the call to include dim=X as an argument.".format(name),
        stacklevel=stacklevel,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret

def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    r"""Applies a softmax function.
    Softmax is defined as:
    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`
    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.
    See :class:`~torch.nn.Softmax` for more details.
    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).
    """
    # if has_torch_function_unary(input):
    #     return handle_torch_function(softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Shape:
        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    # if has_torch_function_variadic(input, weight):
    #     return handle_torch_function(linear, (input, weight), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)

def _pad(input: Tensor, pad: List[int], mode: str = "constant", value: float = 0) -> Tensor:
    r"""Pads tensor.
    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding\_left}, \text{padding\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom}`
        :math:`\text{padding\_front}, \text{padding\_back})`.
    Padding mode:
        See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and
        :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the
        padding modes works. Constant padding is implemented for arbitrary dimensions.
        Replicate padding is implemented for padding the last 3 dimensions of 5D input
        tensor, or the last 2 dimensions of 4D input tensor, or the last dimension of
        3D input tensor. Reflect padding is only implemented for padding the last 2
        dimensions of 4D input tensor, or the last dimension of 3D input tensor.
    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.
    Args:
        input (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``
    Examples::
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 9, 7, 3])
    """
    # if has_torch_function_unary(input):
    #     return handle_torch_function(_pad, (input,), input, pad, mode=mode, value=value)
    assert len(pad) % 2 == 0, "Padding length must be divisible by 2"
    assert len(pad) // 2 <= input.dim(), "Padding length too large"
    if mode == "constant":
        return _VF.constant_pad_nd(input, pad, value)
    else:
        assert value == 0, 'Padding mode "{}"" doesn\'t take in value argument'.format(mode)
        if input.dim() == 3:
            assert len(pad) == 2, "3D tensors expect 2 values for padding"
            if mode == "reflect":
                return torch._C._nn.reflection_pad1d(input, pad)
            elif mode == "replicate":
                return torch._C._nn.replication_pad1d(input, pad)
            elif mode == "circular":
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError

        elif input.dim() == 4:
            assert len(pad) == 4, "4D tensors expect 4 values for padding"
            if mode == "reflect":
                return torch._C._nn.reflection_pad2d(input, pad)
            elif mode == "replicate":
                return torch._C._nn.replication_pad2d(input, pad)
            elif mode == "circular":
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError

        elif input.dim() == 5:
            assert len(pad) == 6, "5D tensors expect 6 values for padding"
            if mode == "reflect":
                raise NotImplementedError
            elif mode == "replicate":
                return torch._C._nn.replication_pad3d(input, pad)
            elif mode == "circular":
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("Only 3D, 4D, 5D padding with non-constant padding are supported for now")


# We define this function as _pad because it takes an argument
# named pad, which clobbers the recursive reference to the pad
# function needed for __torch_function__ support
pad = _pad


def _pad_circular(input: Tensor, padding: List[int]) -> Tensor:
    """Circularly pads tensor.
    Tensor values at the beginning are used to pad the end, and values at the
    end are used to pad the beginning. For example, consider a single dimension
    with values [0, 1, 2, 3]. With circular padding of (1, 1) it would be
    padded to [3, 0, 1, 2, 3, 0], and with padding (1, 2) it would be padded to
    [3, 0, 1, 2, 3, 0, 1]. If negative padding is applied then the ends of the
    tensor get removed. With circular padding of (-1, -1) the previous example
    would become [1, 2]. Circular padding of (-1, 1) would produce
    [1, 2, 3, 1].
    The first and second dimensions of the tensor are not padded.
    Args:
        input: Tensor with shape :math:`(N, C, D[, H, W])`.
        padding: Tuple containing the number of elements to pad each side of
            the tensor. The length of padding must be twice the number of
            paddable dimensions. For example, the length of padding should be 4
            for a tensor of shape :math:`(N, C, H, W)`, and the length should
            be 6 for a tensor of shape :math:`(N, C, D, H, W)`.
    Examples::
        >>> x = torch.tensor([[[[0, 1, 2], [3, 4, 5]]]])  # Create tensor
        >>> # Example 1
        >>> padding = (1, 1, 1, 1)
        >>> y = F.pad(x, padding, mode='circular')
        >>> print(y)
        tensor([[[[5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0]]]])
        >>> print(y.shape)
        torch.Size([1, 1, 4, 5])
        >>> # Example 2
        >>> padding = (1, 1, 2, 2)
        >>> z = F.pad(x, padding, mode='circular')
        >>> print(z)
        tensor([[[[2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3],
                  [2, 0, 1, 2, 0],
                  [5, 3, 4, 5, 3]]]])
        >>> print(z.shape)
        torch.Size([1, 1, 6, 5])
    """
    in_shape = input.shape
    paddable_shape = in_shape[2:]
    ndim = len(paddable_shape)

    for idx, size in enumerate(paddable_shape):
        # Only supports wrapping around once
        assert padding[-(idx * 2 + 1)] <= size, "Padding value causes wrapping around more than once."
        assert padding[-(idx * 2 + 2)] <= size, "Padding value causes wrapping around more than once."
        # Negative padding should not result in negative sizes
        assert (
            padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)] + size >= 0
        ), "Negative padding value is resulting in an empty dimension."

    # Get shape of padded tensor
    out_shape = in_shape[:2]
    for idx, size in enumerate(paddable_shape):
        out_shape += (size + padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)],)

    out = torch.empty(out_shape, dtype=input.dtype, layout=input.layout, device=input.device)

    # Put original array in padded array
    if ndim == 1:
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        out[..., out_d0:out_d1] = input[..., in_d0:in_d1]
    elif ndim == 2:
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        out_h0 = max(padding[-4], 0)
        out_h1 = out_shape[3] - max(padding[-3], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        in_h0 = max(-padding[-4], 0)
        in_h1 = in_shape[3] - max(-padding[-3], 0)

        out[..., out_d0:out_d1, out_h0:out_h1] = input[..., in_d0:in_d1, in_h0:in_h1]
    elif ndim == 3:
        out_d0 = max(padding[-2], 0)
        out_d1 = out_shape[2] - max(padding[-1], 0)

        out_h0 = max(padding[-4], 0)
        out_h1 = out_shape[3] - max(padding[-3], 0)

        out_w0 = max(padding[-6], 0)
        out_w1 = out_shape[4] - max(padding[-5], 0)

        in_d0 = max(-padding[-2], 0)
        in_d1 = in_shape[2] - max(-padding[-1], 0)

        in_h0 = max(-padding[-4], 0)
        in_h1 = in_shape[3] - max(-padding[-3], 0)

        in_w0 = max(-padding[-6], 0)
        in_w1 = in_shape[4] - max(-padding[-5], 0)

        out[..., out_d0:out_d1, out_h0:out_h1, out_w0:out_w1] = input[..., in_d0:in_d1, in_h0:in_h1, in_w0:in_w1]

    # The following steps first pad the beginning of the tensor (left side),
    # and then pad the end of the tensor (right side).
    # Note: Corners will be written more than once when ndim > 1.

    # Only in cases where padding values are > 0 are when additional copying
    # is required.

    # Pad first dimension (depth)
    if padding[-2] > 0:
        i0 = out_shape[2] - padding[-2] - max(padding[-1], 0)
        i1 = out_shape[2] - max(padding[-1], 0)
        o0 = 0
        o1 = padding[-2]
        out[:, :, o0:o1] = out[:, :, i0:i1]
    if padding[-1] > 0:
        i0 = max(padding[-2], 0)
        i1 = max(padding[-2], 0) + padding[-1]
        o0 = out_shape[2] - padding[-1]
        o1 = out_shape[2]
        out[:, :, o0:o1] = out[:, :, i0:i1]

    # Pad second dimension (height)
    if len(padding) > 2:
        if padding[-4] > 0:
            i0 = out_shape[3] - padding[-4] - max(padding[-3], 0)
            i1 = out_shape[3] - max(padding[-3], 0)
            o0 = 0
            o1 = padding[-4]
            out[:, :, :, o0:o1] = out[:, :, :, i0:i1]
        if padding[-3] > 0:
            i0 = max(padding[-4], 0)
            i1 = max(padding[-4], 0) + padding[-3]
            o0 = out_shape[3] - padding[-3]
            o1 = out_shape[3]
            out[:, :, :, o0:o1] = out[:, :, :, i0:i1]

    # Pad third dimension (width)
    if len(padding) > 4:
        if padding[-6] > 0:
            i0 = out_shape[4] - padding[-6] - max(padding[-5], 0)
            i1 = out_shape[4] - max(padding[-5], 0)
            o0 = 0
            o1 = padding[-6]
            out[:, :, :, :, o0:o1] = out[:, :, :, :, i0:i1]
        if padding[-5] > 0:
            i0 = max(padding[-6], 0)
            i1 = max(padding[-6], 0) + padding[-5]
            o0 = out_shape[4] - padding[-5]
            o1 = out_shape[4]
            out[:, :, :, :, o0:o1] = out[:, :, :, :, i0:i1]

    return out


def my_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            my_multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None