import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        the_dict = pickle.load(fo, encoding='bytes')
    return the_dict


def get_training_set(i):
    the_dict = unpickle('cifar-10-batches-py/data_batch_' + str(i))
    x = the_dict[b'data']
    x = format_x_data(x)
    y = the_dict[b'labels']
    return x, y


def get_dev_set():
    the_dict = unpickle('cifar-10-batches-py/test_batch')
    x = the_dict[b'data'][:5000]
    x = format_x_data(x)
    y = the_dict[b'labels'][:5000]
    return x, y


def get_test_set():
    the_dict = unpickle('cifar-10-batches-py/test_batch')
    x = the_dict[b'data'][5000:]
    x = format_x_data(x)
    y = the_dict[b'labels'][5000:]
    return x, y


def format_x_data(x):
    return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
