import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from typing import *


def check_tensor_dep_free(obj):
    """check if `x` is torch.Tensor but dependency-free"""
    return type(obj).__module__.startswith('torch') and 'Tensor' in type(obj).__name__


class PreProcessor:
    def __init__(
        self,
    ):
        pass
    
    def __call__(
        self, inputs: List[object] | List[List[object]]
    ) -> List[List[object]]:
        # step.0 -> all obj to numpy
        if check_tensor_dep_free(inputs):
            inputs = inputs.cpu().numpy()
        elif isinstance(inputs, np.ndarray):
            pass
        elif isinstance(inputs, plt.Figure):
            inputs = [[self.parse_figure_obj(inputs)]]
        elif isinstance(inputs, list):
            for i, row in enumerate(inputs):
                if check_tensor_dep_free(row):
                    inputs[i] = row.cpu().numpy()
                elif isinstance(row, plt.Figure):
                    inputs[i] = [self.parse_figure_obj(row)]    
                elif isinstance(row, list):
                    for j, obj in enumerate(row):
                        if check_tensor_dep_free(obj):
                            inputs[i][j] = obj.cpu().numpy()
        else:
            raise ValueError(f'Unexpected input type: {type(inputs)}')
        
        # step.1 -> split batch object if need.
        if isinstance(inputs, np.ndarray):
            # case.0 - 2D
            if len(inputs.shape) == 2:
                inputs = inputs[None]
            # case.1 - 3D: (C, H, W) / (H, W, C), (B, H, W)
            if len(inputs.shape) == 3:
                if inputs.shape[0] in [1, 3] or inputs.shape[-1] in [1, 3]:
                    inputs = [[inputs]]
                else:
                    inputs = inputs[:, None]
            # case.2 - 4D: (B, ...)
            if hasattr(inputs, 'shape') and len(inputs.shape) == 4:
                inputs = [inputs[i] for i in range(inputs.shape[0])]
            # case.3 - invalid.
            elif isinstance(inputs, np.ndarray): 
                raise ValueError(f'Unexpected input with shape: {inputs.shape}')
        
        elif isinstance(inputs, list):
            if not isinstance(inputs[0], list): inputs = [inputs]  # to list[list]
            
            for i, row in enumerate(inputs):
                if isinstance(row, np.ndarray):
                    if len(row.shape) == 2:
                        row = row[None]
                    if len(row.shape) == 3:
                        if row.shape[0] in [1, 3] or row.shape[-1] in [1, 3]:
                            inputs[i] = [row, ...] 
                            continue
                        else:
                            row = row[:, None]
                    if hasattr(row, 'shape') and len(row.shape) == 4:
                        print(row.shape)
                        inputs[i] = [row[i] for i in range(row.shape[0])] + [...]
                        print(inputs[i])
                    elif isinstance(row, np.ndarray): 
                        raise ValueError(f'Unexpected input with shape: {row.shape}')

        # step.2 -> parse special elements
        num_row_objs = max([len(row) - sum(obj is ... for obj in row) for row in inputs])
        
                
        for i, row in enumerate(inputs):
            for j, obj in enumerate(row):
                if obj is ...: 
                    inputs[i] = self.parse_ellipsis(row, num_row_objs)
                    break
                elif obj is None: 
                    inputs[i][j] = self.parse_none()
        
        assert len(set([len(row) for row in inputs])) == 1, \
            f'Inconsistent number of objects between rows. Use `...` or `None` to set/skip null objects.'
        
        # step.3 -> convert objects
        for i, row in enumerate(inputs):
            for j, obj in enumerate(row):
                obj = self.to_numpy(obj)
                obj = self.to_uint8(obj)
                obj = self.to_hwc_fmt(obj)
                obj = self.to_three_channels(obj)
                inputs[i][j] = obj
        
        return inputs
    
    def to_numpy(self, x):
        if isinstance(x, plt.Figure):
            return self.parse_figure_obj(x)
        else:
            return x
    
    def to_uint8(self, x):
        if isinstance(x, NullObject) or x.dtype is np.uint8:
            return x
        elif x.dtype is np.bool:
            return (x * 255).astype(np.uint8)
        
        min_val, max_val = x.min(), x.max()
        
        # case.0 - (-1, 1)
        if -1 <= min_val < 0 and 0 < max_val <= 1:
            return ((x + 1) * 127.5).astype(np.uint8)
        elif 0 <= min_val < 1 and 0 <= max_val <= 1:
            return (x * 127.5).astype(np.uint8)
        else:
            return ((x - min_val) / (max_val - min_val)).astype(np.uint8)
    
    def to_hwc_fmt(self, x):
        if isinstance(x, NullObject):
            return x

        if len(x.shape) == 2:
            return x[..., None]
        elif len(x.shape) == 3:
            if x.shape[0] in [1, 3]:
                return np.transpose(x, (1, 2, 0))
            if x.shape[-1] not in [1, 3]:
                raise ValueError(f'Unexpected target with unclear channels: {x.shape[-1]}.')
            else:
                return x
        raise ValueError(f'>3D Object should not be mixed with other objects. Please split or place it in new line.')
    
    def to_three_channels(self, x):
        if isinstance(x, NullObject) or x.shape[-1] == 3:
            return x
        
        if x.shape[-1] == 1:
            return np.repeat(x, axis=-1, repeats=3)
    
    def parse_ellipsis(self, row, row_length):
        assert not any([obj is ... for obj in row[1:-1]]), f'Found `...` inside the row, which is ambiguous.'
        assert not (row[0] is ... and row[-1] is ...), 'Found multiple `...` in one row, which is ambiguous.'
        
        if len(row) - 1 == row_length:
            fillings = []
        else:
            fillings = [self.parse_none() for _ in range(max(0, row_length + 1 - len(row)))]
        
        return fillings + row[1:] if row[0] is ... else row[:-1] + fillings
        
    def parse_none(self):
        return NullObject()
    
    def parse_figure_obj(self, fig):
        fig.canvas.draw()
        rgba_buf = fig.canvas.buffer_rgba()
        (w, h) = fig.canvas.get_width_height()
        rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        return rgba_arr


class NullObject:
    def __init__(self):
        self.shape = (-1,)
    
    def __str__(self):
        return 'null_object'