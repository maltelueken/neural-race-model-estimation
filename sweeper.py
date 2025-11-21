def constant_list(length, value):
    return (value,) * length

def get_decay_steps(num_epochs, num_batches):
    return num_epochs * num_batches