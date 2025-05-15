class ndarray:
    def __init__(self, data):
        self.data = data
        self.shape = self._get_shape(data)

    def _get_shape(self, data):
        if isinstance(data, list):
            return (len(data),) + self._get_shape(data[0]) if data else ()
        return ()

    def __getitem__(self, index):
        return self._recursive_getitem(self.data, index)

    def _recursive_getitem(self, data, index):
        if isinstance(index, tuple):
            if len(index) == 1:
                return self._recursive_getitem(data, index[0])
            return self._recursive_getitem(data[index[0]], index[1:])
        return data[index]

    def __setitem__(self, index, value):
        self._recursive_setitem(self.data, index, value)

    def _recursive_setitem(self, data, index, value):
        if isinstance(index, tuple):
            if len(index) == 1:
                data[index[0]] = value
            else:
                self._recursive_setitem(data[index[0]], index[1:], value)
        else:
            data[index] = value

    def __repr__(self):
        return f"ndarray({self.data})"

    def reshape(self, new_shape):
        flat_data = self._flatten(self.data)
        if self._calc_size(new_shape) != len(flat_data):
            raise ValueError("cannot reshape array of size {} into shape {}".format(len(flat_data), new_shape))
        return ndarray(self._unflatten(flat_data, new_shape))

    def _flatten(self, data):
        if isinstance(data, list):
            return [item for sublist in data for item in self._flatten(sublist)]
        return [data]

    def _unflatten(self, flat_data, shape):
        if not shape:
            return flat_data[0]
        size = shape[0]
        rest_shape = shape[1:]
        return [self._unflatten(flat_data[i * self._calc_size(rest_shape):(i + 1) * self._calc_size(rest_shape)], rest_shape) for i in range(size)]

    def _calc_size(self, shape):
        size = 1
        for dim in shape:
            size *= dim
        return size

    def _broadcast_shape(self, shape1, shape2):
        if len(shape1) < len(shape2):
            shape1 = (1,) * (len(shape2) - len(shape1)) + shape1
        elif len(shape2) < len(shape1):
            shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
        result_shape = []
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 == dim2 or dim1 == 1 or dim2 == 1:
                result_shape.append(max(dim1, dim2))
            else:
                raise ValueError("shapes {} and {} are not broadcastable".format(shape1, shape2))
        return tuple(result_shape)

    def _broadcast_to_shape(self, data, shape):
        current_shape = self._get_shape(data)
        if current_shape == shape:
            return data
        if not current_shape:
            return data * self._calc_size(shape)
        result = []
        for i in range(shape[0]):
            result.append(self._broadcast_to_shape(data[i % len(data)], shape[1:]))
        return result

    def _apply_elementwise_operation(self, other, operation):
        if isinstance(other, (ndarray, list)):
            result_shape = self._broadcast_shape(self.shape, ndarray(other).shape if isinstance(other, list) else other.shape)
            self_data = self._broadcast_to_shape(self.data, result_shape)
            other_data = self._broadcast_to_shape(other.data if isinstance(other, ndarray) else other, result_shape)
        else:
            result_shape = self.shape
            self_data = self.data
            other_data = other

        result_data = self._recursive_apply_operation(self_data, other_data, operation)
        return ndarray(result_data)

    def _recursive_apply_operation(self, data1, data2, operation):
        if isinstance(data1, list) and isinstance(data2, list):
            return [self._recursive_apply_operation(d1, d2, operation) for d1, d2 in zip(data1, data2)]
        if isinstance(data1, list):
            return [self._recursive_apply_operation(d1, data2, operation) for d1 in data1]
        if isinstance(data2, list):
            return [self._recursive_apply_operation(data1, d2, operation) for d2 in data2]
        return operation(data1, data2)

    def __add__(self, other):
        return self._apply_elementwise_operation(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._apply_elementwise_operation(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._apply_elementwise_operation(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._apply_elementwise_operation(other, lambda x, y: x / y)

    def count_nonzero(self):
        return self._recursive_count_nonzero(self.data)

    def _recursive_count_nonzero(self, data):
        if isinstance(data, list):
            return sum(self._recursive_count_nonzero(item) for item in data)
        return 1 if data != 0 else 0

def array(data):
    return ndarray(data)

def zeros(shape):
    if not shape:
        return 0  # return scalar zero if shape is empty
    # Recursive function to build nested list with zeros
    def build_zeros(curr_shape):
        if len(curr_shape) == 1:
            return [0] * curr_shape[0]
        return [build_zeros(curr_shape[1:]) for _ in range(curr_shape[0])]

    data = build_zeros(shape)
    return ndarray(data)

def count_nonzero(input: ndarray):
    return input.count_nonzero()
