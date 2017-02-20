import tensorflow as tf, numpy as np

def create(config):
	dim_b, dim_i, dim_f, dim_s, dim_d, dim_o = config.getint('batch'), config.getint('inputs'), config.getint('slice'), config.getint('dice'), config.getint('layers'), config.getint('outputs')
	cwidth, cstride, pwidth, pstride = config.getint('cwidth'), config.getint('cstride'), config.getint('pwidth'), config.getint('pstride')
	lrate, dstep, drate, optim, rfact = config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim')), config.getfloat('rfact')
	pool, nonlinear, reg = getattr(tf.nn, config.get('pool')), getattr(tf.nn, config.get('nonlinear')), getattr(tf.contrib.layers, config.get('reg'))
	model = dict()

	model['o'] = tf.placeholder(tf.float32, [dim_b, dim_o], name = 'o')
	model['x'] = tf.placeholder(tf.float32, [dim_b, dim_i], name = 'x')
	model['xc'] = tf.reshape(tf.slice(model['x'], [0, 0], [dim_b, dim_f]), [dim_b, dim_f / dim_s, dim_s], name = 'xc')
	model['xf'] = tf.slice(model['x'], [0, dim_f], [dim_b, dim_i - dim_f], name = 'xf')
	dim_x, dim_y = dim_f / dim_s, dim_s
	for i in xrange(dim_d):
		model['Wc_%i' %i] = tf.Variable(tf.random_uniform([cwidth, cwidth, 1, 1], -np.sqrt(3. / (cwidth)), np.sqrt(3. / (cwidth))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'Wc_%i' %i)
		model['xc_%i' %i] = tf.expand_dims(model['xc'], -1, name = 'xc_%i' %i) if i == 0 else model['yc_%i' %(i - 1)]
		model['yc_%i' %i] = pool(tf.nn.conv2d(model['xc_%i' %i], model['Wc_%i' %i], [1, cstride, cstride, 1], 'VALID'), [1, pwidth, pwidth, 1], [1, pstride, pstride, 1], 'VALID', name = 'yc_%i' %i)
		dim_x = int(np.ceil(float(int(np.ceil(float(dim_x - cwidth + 1) / float(cstride))) - pwidth + 1) / float(pstride)))
		dim_y = int(np.ceil(float(int(np.ceil(float(dim_y - cwidth + 1) / float(cstride))) - pwidth + 1) / float(pstride)))
		model['Wf_%i' %i] = tf.Variable(tf.random_uniform([dim_i - dim_f, dim_i - dim_f], -np.sqrt(3. / (dim_i - dim_f)), np.sqrt(3. / (dim_i - dim_f))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'Wf_%i' %i)
		model['bf_%i' %i] = tf.Variable(tf.random_uniform([1, dim_i - dim_f], -np.sqrt(6. / (dim_i - dim_f)), np.sqrt(6. / (dim_i - dim_f))), name = 'bf_%i' %i)
		model['xf_%i' %i] = model['xf'] if i == 0 else model['yc_%i' %(i - 1)]
		model['yf_%i' %i] = nonlinear(tf.add(tf.matmul(model['xf_%i' %i], model['Wf_%i' %i]), model['bf_%i' %i]), name = 'yf_%i' %i)
	model['W_%i' %dim_d] = tf.Variable(tf.random_uniform([dim_x * dim_y + dim_i - dim_f, dim_o], -np.sqrt(6. / (dim_x * dim_y + dim_i - dim_f + dim_o)), np.sqrt(6. / (dim_x * dim_y + dim_i - dim_f + dim_o))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'W_%i' %dim_d)
	model['b_%i' %dim_d] = tf.Variable(tf.random_uniform([1, dim_o], -np.sqrt(6. / dim_o), np.sqrt(6. / dim_o)), name = 'b_%i' %dim_d)
	model['x_%i' %dim_d] = tf.concat([tf.reshape(tf.squeeze(model['yc_%i' %(dim_d - 1)], [-1]), [dim_b, dim_x * dim_y]), model['yf_%i' %(dim_d - 1)]], 1, name = 'xc_%i' %dim_d)
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
