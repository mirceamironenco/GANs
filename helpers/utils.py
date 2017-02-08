def print_flags(flags):
	for key, value in vars(flags).items():
		print("{}: {}".format(key, str(value)))

def load_mnist():
	from tensorflow.examples.tutorials.mnist import input_data
	return input_data.read_data_sets("./MNIST_data", one_hot=True)
