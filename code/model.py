import sys, configparser, datetime
import dense, conv
import tensorflow as tf, numpy as np

def feed(model, config, filename):
	featurelist, labellist, batch, labels = list(), list(), config.getint('global', 'batch'), config.getint('global', 'outputs')
	for line in open(filename):
		split = line.split()
		featurelist.append([float(feature.split(':')[1]) for feature in split[2: ]])
		label = [1. if i == int(split[0]) else 0. for i in xrange(labels)]
		labellist.append(label)
		if len(labellist) == batch:
			feeddict = dict()
			feeddict.update({model['x']: featurelist})
			feeddict.update({model['o']: labellist})
			yield feeddict
			featurelist, labellist = list(), list()

def run(model, config, session, summary, filename, train):
	iters, freq, saves, total = config.getint('global', 'epochs') if train else 1, config.getint('global', 'print'), config.get('global', 'output'), 0.
	for i in xrange(iters):
		for ii, feeddict in enumerate(feed(model, config, filename)):
			if train:
				val, train = session.run([model['nll'], model['train']], feed_dict = feeddict)
				total += val
				if (ii + 1) % freq == 0:
					summ = session.run(model['nlls'], feed_dict = feeddict)
					summary.add_summary(summ, model['gsd'].eval())
					print datetime.datetime.now(), 'iteration', i, 'batch', ii, 'loss:', val, total
			else:
				vals, exps = session.run(model['p'], feed_dict = feeddict), np.argmax(feeddict[model['o']], 1)
				for exp, val in zip(exps, vals):
					if exp == val: total += 1
	return total

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])

	print datetime.datetime.now(), 'creating model'
	metamodel = config.get('global', 'model')
	model = globals()[metamodel].create(config[metamodel])

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		tf.train.Saver().restore(sess, config.get('global', 'load'))
		summary = tf.train.SummaryWriter(config.get('global', 'logs'), sess.graph)

		print datetime.datetime.now(), 'training model'
		trainingloss = run(model, config, sess, summary, '%s/train' %config.get('global', 'data'), True)
		print datetime.datetime.now(), 'training loss', trainingloss
		print datetime.datetime.now(), 'saving model'
		tf.train.Saver().save(sess, config.get('global', 'save'))
		print datetime.datetime.now(), 'testing model'
		testingaccuracy = run(model, config, sess, summary, '%s/dev' %config.get('global', 'data'), False)
		print datetime.datetime.now(), 'testing accuracy', testingaccuracy
