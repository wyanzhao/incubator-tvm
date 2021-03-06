# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin, no-else-return
"""Conv3D operators"""
from tvm import te

from .pad import pad
from .util import get_pad_tuple3d
from ..util import simplify


def conv3d_ncdhw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Conv3D operator in NCDHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    Filter : tvm.te.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    stride : int or a list/tuple of three ints
        Stride size, or [strid_depth, stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    batch, in_channel, in_depth, in_height, in_width = Input.shape
    num_filter, channel, kernel_d, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_depth = simplify((in_depth - dilated_kernel_d + pad_front + pad_back) // stride_d + 1)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_front, pad_top, pad_left]
    pad_after = [0, 0, pad_back, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = te.reduce_axis((0, in_channel), name='rc')
    rz = te.reduce_axis((0, kernel_d), name='rz')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')

    return te.compute(
        (batch, out_channel, out_depth, out_height, out_width),
        lambda nn, ff, zz, yy, xx: te.sum(
            temp[nn, rc, zz * stride_d + rz * dilation_d, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, rz, ry, rx].astype(out_dtype),
            axis=[rc, rz, ry, rx]), tag="conv3d_ncdhw")


def conv3d_ndhwc(Input, Filter, stride, padding, dilation, out_dtype='float32'):
    """Convolution operator in NDHWC layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        5-D with shape [batch, in_depth, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        5-D with shape [filter_depth, filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of three ints
        Stride size, or [stride_depth, stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_depth, out_height, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3

    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    batch, in_depth, in_height, in_width, in_channel = Input.shape
    kernel_d, kernel_h, kernel_w, channel, num_filter = Filter.shape
    # compute the output shape
    dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_depth = simplify((in_depth - dilated_kernel_d + pad_front + pad_back) // stride_d + 1)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_front, pad_top, pad_left, 0]
    pad_after = [0, pad_back, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rd = te.reduce_axis((0, kernel_d), name='rd')
    rh = te.reduce_axis((0, kernel_h), name='rh')
    rw = te.reduce_axis((0, kernel_w), name='rw')
    rc = te.reduce_axis((0, in_channel), name='rc')
    Output = te.compute(
        (batch, out_depth, out_height, out_width, out_channel),
        lambda nn, dd, hh, ww, cc: te.sum(
            PaddedInput[nn, dd * stride_d + rd * dilation_d, hh * stride_h + rh * dilation_h,
                        ww * stride_w + rw * dilation_w, rc].astype(out_dtype) *
            Filter[rd, rh, rw, rc, cc].astype(out_dtype), axis=[rd, rh, rw, rc]),
        name="Conv3dOutput", tag="conv3d_ndhwc")
    return Output
