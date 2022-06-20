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
# pylint: disable=too-many-arguments
"""Argsort operator"""
import tvm
from tvm import te
from .utils import get_const_tuple
from ..tir import ir_builder
from .math import cast

def sort(data, axis=-1, is_ascend=1):
    """Performs sorting along the given axis and returns an array
    in sorted order.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    axis : int, optional
        Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        Sorted index tensor.

    """
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    out_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "out_buf", data_alignment=8)
    out = te.extern(
        data.shape,
        [data],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.sort.sort", ins[0], outs[0], axis, is_ascend
        ),
        dtype=data.dtype,
        in_buffers=[data_buf],
        out_buffers=out_buf,
        name="sort_cpu",
        tag="sort_cpu",
    )
    return out

def argsort_nms_te_(data, valid_count, out, axis, is_ascend):
    '''
     Very naive, very ugly implementation of an argsort
     TODO: improve this!
    '''
    #breakpoint()
    # TODO (FP): implement the use of valid_count!
    # TODO (FP): implement the use of is_ascend!

    ib = ir_builder.create()

    data_shape = data.shape
    data_dtype = data.dtype

    out = ib.buffer_ptr(out)
    data = ib.buffer_ptr(data)
    valid_count = ib.buffer_ptr(valid_count)

    lo = ib.allocate("int32", (1,), name="lo", scope="local")
    hi = ib.allocate("int32", (1,), name="hi", scope="local")
    lo_inner = ib.allocate("int32", (1,), name="lo_inner", scope="local")

    current_max = ib.allocate(data_dtype, (1,), name="current_max", scope="local")
    current_min = ib.allocate(data_dtype, (1,), name="current_min", scope="local")
    current_sortedindex = ib.allocate("int32", (1,), name="current_sortedindex", scope="local")

    hi[0] = cast(data_shape[axis], "int32")
    current_sortedindex[0] = cast(0, "int32")

    if not is_ascend:
        # First, get the maximum value
        current_max[0] = data[0,lo[0]]
        lo[0] = cast(1, "int32")
        with ib.while_loop(lo[0] < hi[0]):
            with ib.if_scope(data[0,lo[0]] > current_max[0]):
                current_max[0] = data[0,lo[0]]
                current_sortedindex[0] = lo[0]
            lo[0] = lo[0] + 1
    else:
        # First, get the minimum value
        current_min[0] = cast(0x7fffffff, "float32")
        lo[0] = cast(0, "int32")
        with ib.while_loop(lo[0] < hi[0]):
            with ib.if_scope(data[0,lo[0]] < current_min[0]):
                current_min[0] = data[0,lo[0]]
                current_sortedindex[0] = lo[0]
            lo[0] = lo[0] + 1

    # Insert this into the first position of the output
    out[0,0] = current_sortedindex[0]

    # Now get all the other indexes
    lo[0] = cast(1, "int32")
    with ib.while_loop(lo[0] < hi[0]):
        lo_inner[0] = cast(0, "int32")
        current_max[0] = cast(0, data_dtype)
        current_min[0] = cast(0x7fffffff, "float32")
        with ib.while_loop(lo_inner[0] < hi[0]):
            if not is_ascend:
                with ib.if_scope(data[0,lo_inner[0]] > current_max[0]):
                    with ib.if_scope(data[0,lo_inner[0]] >= data[0,out[0,lo[0]-1]]):
                        current_sortedindex[0] = lo_inner[0]
                        current_max[0] = data[0,lo_inner[0]]
            else:
                with ib.if_scope(data[0,lo_inner[0]] < current_min[0]):
                    with ib.if_scope(data[1,lo_inner[0]] <= data[0,out[0,lo[0]-1]]):
                        current_sortedindex[0] = lo_inner[0]
                        current_min[0] = data[0,lo_inner[0]]
            lo_inner[0] = lo_inner[0] + 1

        out[0,lo[0]] = current_sortedindex[0]
        with ib.if_scope(valid_count[0] <= lo[0]):
            # If we already sorted the amount of data required by valid_count, finish loop
            lo[0] = hi[0]
        with ib.else_scope():
            # Else, continue
            lo[0] = lo[0] + 1

    #out = te.compute(
    #    data.shape, 
    #    lambda i,j: data[i,j].astype("int32")
    #)
    return ib.get()

def argsort_te(ib,sorter,sorter_size,is_ascend):
    lo = ib.allocate("int32", (1,), name="lo", scope="local")
    lo_inner = ib.allocate("int32", (1,), name="lo_inner", scope="local")
    hi = ib.allocate("int32", (1,), name="hi", scope="local")
    current_sortedindex = ib.allocate("int32", (1,), name="current_sortedindex", scope="local")
    sorted_indexes = ib.allocate("int32", (sorter_size,), name="current_sortedindex", scope="local")

    current_max = ib.allocate("float32", (1,), name="current_max", scope="local")
    current_min = ib.allocate("float32", (1,), name="current_min", scope="local")

    if not is_ascend:
        # First, get the maximum value
        current_max[0] = sorter[0]
        current_sortedindex[0] = cast(0, "int32")
        lo[0] = cast(1, "int32")
        hi[0] = cast(sorter_size, "int32")
        with ib.while_loop(lo[0] < hi[0]):
            with ib.if_scope(sorter[lo[0]] > current_max[0]):
                current_max[0] = sorter[lo[0]]
                current_sortedindex[0] = lo[0]
            lo[0] = lo[0] + 1
    else:
        # First, get the minimum value
        current_min[0] = cast(0x7fffffff, "float32")
        current_sortedindex[0] = cast(0, "int32")
        lo[0] = cast(0, "int32")
        hi[0] = cast(sorter_size, "int32")
        with ib.while_loop(lo[0] < hi[0]):
            with ib.if_scope(sorter[lo[0]] < current_min[0]):
                current_min[0] = sorter[lo[0]]
                current_sortedindex[0] = lo[0]
            lo[0] = lo[0] + 1

    # Insert this into the first position of the output
    sorted_indexes[0] = current_sortedindex[0]

    # Now get all the other indexes
    lo[0] = cast(1, "int32")
    with ib.while_loop(lo[0] < hi[0]):
        lo_inner[0] = cast(0, "int32")
        current_max[0] = cast(0, "float")
        current_min[0] = cast(0x7fffffff, "float")
        current_sortedindex[0] = cast(0, "int32")
        with ib.while_loop(lo_inner[0] < hi[0]):
            if not is_ascend:
                with ib.if_scope(sorter[lo_inner[0]] > current_max[0]):
                    with ib.if_scope(sorter[lo_inner[0]] < sorter[sorted_indexes[lo[0]-1]]):
                        current_sortedindex[0] = lo_inner[0]
                        current_max[0] = sorter[lo_inner[0]]
            else:
                with ib.if_scope(sorter[lo_inner[0]] < current_min[0]):
                    with ib.if_scope(sorter[lo_inner[0]] > sorter[sorted_indexes[lo[0]-1]]):
                        current_sortedindex[0] = lo_inner[0]
                        current_min[0] = sorter[lo_inner[0]]

            lo_inner[0] = lo_inner[0] + 1

        sorted_indexes[lo[0]] = current_sortedindex[0]
        lo[0] = lo[0] + 1

    return sorted_indexes

def argsort_nms_te(data, valid_count, out, axis, is_ascend):
    '''
     Very naive, very ugly implementation of an argsort
     TODO: improve this!
    '''
    #breakpoint()
    # TODO (FP): implement the use of valid_count!
    # TODO (FP): implement the use of is_ascend!

    ib = ir_builder.create()

    data_shape = data.shape
    data_dtype = data.dtype

    out = ib.buffer_ptr(out)
    data = ib.buffer_ptr(data)
    valid_count = ib.buffer_ptr(valid_count)

    lo = ib.allocate("int32", (1,), name="lo", scope="local")
    hi = ib.allocate("int32", (1,), name="hi", scope="local")
    lo_inner = ib.allocate("int32", (1,), name="lo_inner", scope="local")
    axis_buff = ib.allocate("int32", (1,), name="axis_buff", scope="local")
    axis_buff[0] = axis
    axis_mul_before = ib.allocate("int32", (1,), name="axis_mul_before", scope="local")
    axis_mul_after = ib.allocate("int32", (1,), name="axis_mul_after", scope="local")

    i = ib.allocate("int32", (1,), name="i", scope="local")
    j = ib.allocate("int32", (1,), name="j", scope="local")
    k = ib.allocate("int32", (1,), name="k", scope="local")
    current_sort_num = ib.allocate("int32", (1,), name="current_sort_num", scope="local")
    base_idx = ib.allocate("int32", (1,), name="base_idx", scope="local")
    sorter = ib.allocate("float32", (data_shape[axis],), name="sorter", scope="local")

    #current_max = ib.allocate(data_dtype, (1,), name="current_max", scope="local")
    #current_min = ib.allocate(data_dtype, (1,), name="current_min", scope="local")
    #current_sortedindex = ib.allocate("int32", (1,), name="current_sortedindex", scope="local")

    #hi[0] = cast(data_shape[axis], "int32")
    #current_sortedindex[0] = cast(0, "int32")

    # TODO (Improve this!)
    #axis_mul_before[0] = cast(1, "int32")
    #axis_mul_after[0] = cast(1, "int32")

    mul_bef = 1
    mul_aft = 1
    for i_dim in range(len(data_shape)):
        if i_dim < axis:
            mul_bef *= data_shape[i_dim]
        elif i_dim > axis:
            mul_aft *= data_shape[i_dim]
        
    axis_mul_before[0] = mul_bef
    axis_mul_after[0] = mul_aft

    i[0] = cast(0, "int32")

    with ib.while_loop(i[0] < axis_mul_before[0]):
        j[0] = cast(0, "int32")
        with ib.while_loop(j[0] < axis_mul_after[0]):
            # Clean sorter
            k[0] = cast(0, "int32")
            with ib.while_loop(k[0] < data_shape[axis]):
                sorter[k[0]] = cast(0, "float32")
                k[0] += 1

            # Fill sorter
            current_sort_num[0] = valid_count[i[0]*axis_mul_after[0] + j[0]]
            base_idx = i[0] * data_shape[axis] * axis_mul_after[0] + j[0]
            k[0] = cast(0, "int32")
            with ib.while_loop(k[0] < current_sort_num[0]):
                sorter[k[0]] = data[base_idx + k[0]*axis_mul_after[0]]
                k[0] += 1

            # Actual sort
            sorted_indexes = argsort_te(ib,sorter,data_shape[axis],is_ascend)

            # Assign to output
            k[0] = cast(0, "int32")
            with ib.while_loop(k[0] < data_shape[axis]):
                with ib.if_scope(current_sort_num[0] > k[0]):
                    out[base_idx + k[0]*axis_mul_after[0]] = sorted_indexes[k[0]]
                with ib.else_scope():
                    out[base_idx + k[0]*axis_mul_after[0]] = k[0]
                k[0] += 1

            j[0] += 1
        i[0] += 1

    return ib.get()
    
def argsort(data, valid_count=None, axis=-1, is_ascend=1, dtype="float32"):
    """Performs sorting along the given axis and returns an array
    of indices having the same shape as an input array that index
    data in sorted order.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    valid_count : tvm.te.Tensor, optional
        1-D tensor for valid number of boxes.

    axis : int, optional
        Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        Sorted index tensor.

    Example
    --------
    .. code-block:: python

        # An example to use argsort
        dshape = (1, 5, 6)
        data = te.placeholder(dshape, name="data")
        axis = 0
        is_ascend = False
        out = argsort(data, axis=axis, is_ascend=is_ascend)
        np_data = np.random.uniform(dshape)
        s = topi.generic.schedule_argsort(out)
        f = tvm.build(s, [data, out], "llvm")
        dev = tvm.cpu()
        tvm_data = tvm.nd.array(np_data, dev)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), dev)
        f(tvm_data, tvm_out)
    """
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    if valid_count is not None:
        valid_count_buf = tvm.tir.decl_buffer(
            valid_count.shape, valid_count.dtype, "valid_count_buf", data_alignment=4
        )
        out_buf = tvm.tir.decl_buffer(data.shape, "int32", "out_buf", data_alignment=8)
        out = te.extern(
            data.shape,
            [data, valid_count],
            lambda ins, outs: 
                argsort_nms_te(ins[0], ins[1], outs[0], axis, is_ascend),
                #tvm.tir.call_packed(
                #    "tvm.contrib.sort.argsort_nms", ins[0], ins[1], outs[0], axis, is_ascend
                #),
            dtype="int32",
            in_buffers=[data_buf, valid_count_buf],
            out_buffers=out_buf,
            name="argsort_nms_cpu",
            tag="argsort_nms_cpu",
        )
    else:
        out_buf = tvm.tir.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)
        #out = argsort_nms_te(data,valid_count,axis,is_ascend)
        out = te.extern(
            data.shape,
            [data],
            lambda ins, outs: 
                #argsort_nms_te(ins[0], ins[1], outs[0], axis, is_ascend),
                tvm.tir.call_packed(
                    "tvm.contrib.sort.argsort", ins[0], outs[0], axis, is_ascend
                ),
            dtype=dtype,
            in_buffers=[data_buf],
            out_buffers=out_buf,
            name="argsort_cpu",
            tag="argsort_cpu",
        )
    return out

def topk(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int64"):
    """Get the top k elements in an input tensor along the given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int or tvm.te.Tensor, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    out : tvm.te.Tensor or List[tvm.te.Tensor]
        The computed result.
    """
    assert ret_type in ["both", "values", "indices"]
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    out_shape = list(get_const_tuple(data.shape))
    kvar = tvm.te.size_var("k")
    if not isinstance(k, int):
        out_shape[axis] = kvar
    elif k >= 1:
        out_shape[axis] = k
    out_bufs = []
    if ret_type in ["both", "values"]:
        out_bufs.append(tvm.tir.decl_buffer(out_shape, data.dtype, "value_buf", data_alignment=8))
    if ret_type in ["both", "indices"]:
        out_bufs.append(tvm.tir.decl_buffer(out_shape, dtype, "indices_buf", data_alignment=8))
    out_shapes = [out_shape] * len(out_bufs)

    kv = kvar if not isinstance(k, int) else k
    out = te.extern(
        out_shapes,
        [data],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.sort.topk", ins[0], *outs, kv, axis, ret_type, is_ascend
        ),
        in_buffers=[data_buf],
        out_buffers=out_bufs,
        name="topk_cpu",
        tag="topk_cpu",
    )
    return out
