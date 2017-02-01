import tensorflow as tf, numpy as np

def create(config):
	dim_b, dim_i, dim_d, dim_o = config.getint('batch'), config.getint('inputs'), config.getint('layers'), config.getint('outputs')
	dims = [dim_i] + [config.getint('layer%i' %(i + 1)) for i in xrange(dim_d)] + [dim_o]
	lrate, dstep, drate, optim, rfact = config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim')), config.getfloat('rfact')
	nonlinear, reg = getattr(tf.nn, config.get('nonlinear')), getattr(tf.contrib.layers, config.get('reg'))
	model = dict()

	model['o'] = tf.placeholder(tf.float32, [dim_b, dim_o], name = 'o')
	model['x'] = tf.placeholder(tf.float32, [dim_b, dim_i], name = 'x')
	for i in xrange(dim_d + 1):
		dim_i, dim_o = dims[i], dims[i + 1]
		model['W_%i' %i] = tf.Variable(tf.random_uniform([dim_i, dim_o], -np.sqrt(6. / (dim_i + dim_o)), np.sqrt(6. / (dim_i + dim_o))), collections = [tf.GraphKeys.VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'W_%i' %i)
		model['b_%i' %i] = tf.Variable(tf.random_uniform([1, dim_o], -np.sqrt(6. / dim_o), np.sqrt(6. / dim_o)), name = 'b_%i' %i)
		model['x_%i' %i] = model['x'] if i == 0 else model['y_%i' %(i - 1)]
		model['y_%i' %i] = nonlinear(tf.add(tf.matmul(model['x_%i' %i], model['W_%i' %i]), model['b_%i' %i]), name = 'y_%i' %i)
	model['y'] = tf.nn.softmax(model['y_%i' %(dim_d)], name = 'y')
	model['p'] = tf.argmax(model['y'], 1, name = 'p')

	model['nll'] = tf.reduce_sum(-tf.mul(model['o'], tf.log(tf.add(model['y'], 1e-6))), name = 'nll')
	model['nlls'] = tf.scalar_summary(model['nll'].name, model['nll'])
	model['gsd'] = tf.Variable(0, trainable = False, name = 'gsd')
	model['lrd'] = tf.train.exponential_decay(lrate, model['gsd'], dstep, drate, staircase = False, name = 'lrd')
	model['reg'] = tf.contrib.layers.apply_regularization(reg(rfact), tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	model['train'] = optim(model['lrd']).minimize(model['nll'] + model['reg'], global_step = model['gsd'], name = 'train')

	return model
