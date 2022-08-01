import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from inspect import signature
from functools import wraps
import heapq
import itertools
import time


def activation_function(act,act_input):
        act_func = None
        if act == "sigmoid":
            act_func = tf.nn.sigmoid(act_input)
        elif act == "tanh":
            act_func = tf.nn.tanh(act_input)
            
        elif act == "relu":
            act_func = tf.nn.relu(act_input)
        
        elif act == "elu":
            act_func = tf.nn.elu(act_input)
           
        elif act == "identity":
            act_func = tf.identity(act_input)
            
        elif act == "softmax":
            act_func = tf.nn.softmax(act_input)
         
        elif act == "selu":
            act_func = tf.nn.selu(act_input) 
        
        else:
            raise NotImplementedError("ERROR")
        return act_func  


def get_data_format(data_format):
    if data_format == "UIRT":
        columns = ["user", "item", "rating", "time"]
        
    elif data_format == "UIR":
        columns = ["user", "item", "rating"]
    
    elif data_format == "UIT":
        columns = ["user", "item", "time"] 
        
    elif data_format == "UI":
        columns = ["user", "item"]    
    
    else:
        raise ValueError("please choose a correct data format. ")
    
    return columns


def csr_to_user_dict(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        if len(value.indices) >= 0:
            train_dict[idx] = value.indices.copy().tolist()
    return train_dict
def categoryuser_list(seq_category):

    category_dict1 = seq_category[::-1]
    b = max(category_dict1, key=category_dict1.count)
    return b
def categoryitemnow_list(category_matrix):
    category_dict2 = []
    for idx, value in enumerate(category_matrix):
        if len(value.indices) >= 0:
            category_dict1 = category_matrix[idx]
            c=category_dict1.data.tolist()
            d = c[::-1]
            b = max(d, key=d.count)
            category_dict2.append(b)
    a = category_dict2
    return category_dict2
def category_itemnow_list(catitem_matrix, catitem_matrixtwo):
    category_dict1 = []
    category_dict2 = []
    category_matrix = catitem_matrix + catitem_matrixtwo
    for idx, value in enumerate(category_matrix):
         if len(value.indices) >= 0:
            category_dict1 = category_matrix[idx]
            if len(category_dict1.data) == 0:
                category_dict1 = [0]
                category_dict2.append(0)
            try:
                category_dict2.append(category_dict1.data[-1])
            except:
                print(idx)
    a = category_dict2
    return category_dict2
def csr_to_category_dict(cat_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    category_dict = {}
    for idx, value in enumerate(cat_matrix):
        if len(value.indices) >= 0:
            category_dict[idx] = value.data.copy().tolist()
    return category_dict
def csr_to_itemtime_dict(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        if len(value.indices) > 0:
            train_dict[idx] = value.indices.copy().tolist()
    return train_dict
def csr_to_item_dict_bytime(time_matrix,train_matrix):
    train_dict = {}
    time_matrix = time_matrix
    user_pos_items = csr_to_itemtime_dict(train_matrix)
    for u, items in user_pos_items.items():
        sorted_items = sorted(items, key=lambda x: time_matrix[u,x])
        train_dict[u] = np.array(sorted_items, dtype=np.int32).tolist()
    a = train_dict

    return train_dict
def csr_to_time_dict(time_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    time_dict = {}
    for idx, value in enumerate(time_matrix):
        if len(value.indices) > 0:
            time_dict[idx] = value.data.copy().tolist()
    return time_dict
def csr_to_user_dict_bytime(time_matrix,train_matrix):
    train_dict = {}
    time_matrix = time_matrix
    user_pos_items = csr_to_user_dict(train_matrix)
    for u, items in user_pos_items.items():
        sorted_items = sorted(items, key=lambda x: time_matrix[u,x])
        train_dict[u] = np.array(sorted_items, dtype=np.int32).tolist()
    a = train_dict
    return train_dict

def get_initializer(init_method, stddev):
        if init_method == 'tnormal':
            return tf.compat.v1.random.truncated_normal(stddev=stddev)
        elif init_method == 'uniform':
            return tf.random_uniform_initializer(-stddev, stddev)
        elif init_method == 'normal':
            return tf.random_normal_initializer(stddev=stddev)
        elif init_method == 'xavier_normal':
            return tf.glorot_normal_initializer()
        elif init_method == 'xavier_uniform':
            return tf.glorot_uniform_initializer()
        elif init_method == 'he_normal':
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)
        elif init_method == 'he_uniform':
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=stddev)  


def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass 


def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    """
    a = np.arange(high)
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


def typeassert(*type_args, **type_kwargs):
    def decorate(func):
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def pad_sequences(sequences, value=0., max_len=None,
                  padding='post', truncating='post', dtype=np.int32):
    """Pads sequences to the same length.

    Args:
        sequences (list): A list of lists, where each element is a sequence.
        value (int or float): Padding value. Defaults to `0.`.
        max_len (int or None): Maximum length of all sequences.
        padding (str): `"pre"` or `"post"`: pad either before or after each
            sequence. Defaults to `post`.
        truncating (str): `"pre"` or `"post"`: remove values from sequences
            larger than `max_len`, either at the beginning or at the end of
            the sequences. Defaults to `post`.
        dtype (int or float): Type of the output sequences. Defaults to `np.int32`.

    Returns:
        np.ndarray: Numpy array with shape `(len(sequences), max_len)`.

    Raises:
        ValueError: If `padding` or `truncating` is not understood.
    """
    if max_len is None:
        max_len = np.max([len(x) for x in sequences])

    x = np.full([len(sequences), max_len], value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def inner_product(a, b, name="inner_product"):
    with tf.name_scope(name=name):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def timer(func):
    """The timer decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("%s function cost: %fs" % (func.__name__, end_time - start_time))
        return result
    return wrapper


def l2_loss(*params):
    return tf.add_n([tf.nn.l2_loss(w) for w in params])


def log_loss(yij, name="log_loss"):
    """ bpr loss
    """
    with tf.name_scope(name):
        return -tf.compat.v1.log_sigmoid(yij)


def dropout_sparse(tf_sp_mat, keep_prob, nnz):
    """Dropout for sparse tensors.
    """
    noise_shape = [nnz]
    random_tensor = tf.random_uniform(noise_shape) + keep_prob
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(tf_sp_mat, dropout_mask)
    scale = 1.0 / keep_prob
    return pre_out * scale
