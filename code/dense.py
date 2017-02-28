import tensorflow as tf, numpy as np

def create(config, weights = None):
	dim_b, dim_i, dim_d, dim_o = config.getint('batch'), config.getint('inputs'), config.getint('layers'), config.getint('outputs')
	dims = [dim_i] + [config.getint('layer%i' %(i + 1)) for i in xrange(dim_d)] + [dim_o]
	lrate, dstep, drate, optim, rfact = config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim')), config.getfloat('rfact')
	nonlinear, reg, lfunc = getattr(tf.nn, config.get('nonlinear')), getattr(tf.contrib.layers, config.get('reg')), config.get('lfunc')
	model, weights = dict(), np.ones([1, config.getint('outputs')], np.float32) if weights is None else weights

	model['o'] = tf.placeholder(tf.float32, [dim_b, dim_o], name = 'o')
	model['x'] = tf.placeholder(tf.float32, [dim_b, dim_i], name = 'x')
	for i in xrange(dim_d + 1):
		dim_i, dim_o = dims[i], dims[i + 1]
		model['W_%i' %i] = tf.Variable(tf.random_uniform([dim_i, dim_o], -np.sqrt(6. / (dim_i + dim_o)), np.sqrt(6. / (dim_i + dim_o))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'W_%i' %i)
		model['b_%i' %i] = tf.Variable(tf.random_uniform([1, dim_o], -np.sqrt(6. / dim_o), np.sqrt(6. / dim_o)), name = 'b_%i' %i)
		model['x_%i' %i] = model['x'] if i == 0 else model['y_%i' %(i - 1)]
		model['y_%i' %i] = nonlinear(tf.add(tf.matmul(model['x_%i' %i], model['W_%i' %i]), model['b_%i' %i]), name = 'y_%i' %i)
	model['s'] = tf.nn.sigmoid(model['y_%i' %(dim_d)], name = 's')
	model['y'] = tf.nn.softmax(model['y_%i' %(dim_d)], name = 'y')
	model['p'] = tf.argmax(model['y'], 1, name = 'p')

	if lfunc == 'nll': model['loss'] = tf.reduce_sum(-tf.multiply(tf.constant(weights, dtype = tf.float32), tf.multiply(model['o'], tf.log(tf.add(model['y'], 1e-6)))), name = 'nll')
	if lfunc == 'cse': model['loss'] = tf.reduce_sum(-tf.multiply(tf.constant(weights, dtype = tf.float32), tf.add(tf.multiply(model['o'], tf.log(tf.add(model['s'], 1e-6))), tf.multiply(tf.subtract(1., model['o']), tf.log(tf.add(tf.subtract(1., model['s']), 1e-6))))), name = 'cse')
	if lfunc == 'fid': model['loss'] = tf.reduce_sum(tf.subtract(1, tf.multiply(tf.constant(weights, dtype = tf.float32), tf.add(tf.sqrt(tf.add(tf.multiply(model['o'], model['s']), 1e-6)), tf.sqrt(tf.add(tf.multiply(tf.subtract(1., model['o']), tf.subtract(1., model['s'])), 1e-6))))), name = 'fid')
	model['summary'] = tf.summary.scalar(model['loss'].name, model['loss'])
	model['gsd'] = tf.Variable(0, trainable = False, name = 'gsd')
	model['lrd'] = tf.train.exponential_decay(lrate, model['gsd'], dstep, drate, staircase = False, name = 'lrd')
	model['reg'] = tf.contrib.layers.apply_regularization(reg(rfact), tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	model['train'] = optim(model['lrd']).minimize(model['loss'] + model['reg'], global_step = model['gsd'], name = 'train')

	return model
