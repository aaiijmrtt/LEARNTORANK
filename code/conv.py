import tensorflow as tf, numpy as np

def create(config):
	dim_b, dim_i, dim_d, dim_o = config.getint('batch'), config.getint('inputs'), config.getint('layers'), config.getint('outputs')
	cwidth, cstride, pwidth, pstride = config.getint('cwidth'), config.getint('cstride'), config.getint('pwidth'), config.getint('pstride')
	lrate, dstep, drate, optim, rfact = config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim')), config.getfloat('rfact')
	pool, nonlinear, reg = getattr(tf.nn, config.get('pool')), getattr(tf.nn, config.get('nonlinear')), getattr(tf.contrib.layers, config.get('reg'))
	model = dict()

	model['o'] = tf.placeholder(tf.float32, [dim_b, dim_o], name = 'o')
	model['x'] = tf.placeholder(tf.float32, [dim_b, dim_i], name = 'x')
	for i in xrange(dim_d):
		model['W_%i' %i] = tf.Variable(tf.random_uniform([cwidth, 1, 1, 1], -np.sqrt(6. / (cwidth)), np.sqrt(6. / (cwidth))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'W_%i' %i)
		model['x_%i' %i] = tf.expand_dims(tf.expand_dims(model['x'], -1), -1, name = 'x_%i' %i) if i == 0 else model['y_%i' %(i - 1)]
		model['y_%i' %i] = pool(tf.nn.conv2d(model['x_%i' %i], model['W_%i' %i], [1, cstride, 1, 1], 'VALID'), [1, pwidth, 1, 1], [1, pstride, 1, 1], 'VALID', name = 'y_%i' %i)
		dim_i = int(np.ceil(float(int(np.ceil(float(dim_i - cwidth + 1) / float(cstride))) - pwidth + 1) / float(pstride)))
	model['W_%i' %dim_d] = tf.Variable(tf.random_uniform([dim_i, dim_o], -np.sqrt(6. / (dim_i + dim_o)), np.sqrt(6. / (dim_i + dim_o))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'W_%i' %dim_d)
	model['b_%i' %dim_d] = tf.Variable(tf.random_uniform([1, dim_o], -np.sqrt(6. / dim_o), np.sqrt(6. / dim_o)), name = 'b_%i' %dim_d)
	model['x_%i' %dim_d] = tf.squeeze(model['y_%i' %(dim_d - 1)], [-2, -1], name = 'x_%i' %dim_d)
	model['y_%i' %dim_d] = nonlinear(tf.add(tf.matmul(model['x_%i' %dim_d], model['W_%i' %dim_d]), model['b_%i' %dim_d]), name = 'y_%i' %dim_d)
	model['y'] = tf.nn.softmax(model['y_%i' %(dim_d)], name = 'y')
	model['p'] = tf.argmax(model['y'], 1, name = 'p')

	model['nll'] = tf.reduce_sum(-tf.multiply(model['o'], tf.log(tf.add(model['y'], 1e-6))), name = 'nll')
	model['nlls'] = tf.summary.scalar(model['nll'].name, model['nll'])
	model['gsd'] = tf.Variable(0, trainable = False, name = 'gsd')
	model['lrd'] = tf.train.exponential_decay(lrate, model['gsd'], dstep, drate, staircase = False, name = 'lrd')
	model['reg'] = tf.contrib.layers.apply_regularization(reg(rfact), tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	model['train'] = optim(model['lrd']).minimize(model['nll'] + model['reg'], global_step = model['gsd'], name = 'train')

	return model
