import sys, configparser, datetime, math, signal
import dense, conv, feat
import tensorflow as tf, numpy as np

def feed(model, config, filename):
	featurelist, labellist, qidlist, batch, labels = list(), list(), list(), config.getint('global', 'batch'), config.getint('global', 'outputs')
	for line in open(filename):
		split = line.split()
		featurelist.append([float(feature.split(':')[1]) for feature in split[2: ]])
		label, qid = [1. if i == int(split[0]) else 0. for i in xrange(labels)], int(split[1].split(':')[1])
		labellist.append(label)
		qidlist.append(qid)
		if len(labellist) == batch:
			feeddict = dict()
			feeddict.update({model['x']: featurelist})
			feeddict.update({model['o']: labellist})
			yield qidlist, feeddict
			featurelist, labellist, qidlist = list(), list(), list()

def run(model, config, session, summary, filename, train):
	iters, freq, saves, total, evallist = config.getint('global', 'epochs') if train else 1, config.getint('global', 'print'), config.get('global', 'output'), 0., list()
	for i in xrange(iters):
		for ii, (qidlist, feeddict) in enumerate(feed(model, config, filename)):
			if train:
				val, _ = session.run([model['nll'], model['train']], feed_dict = feeddict)
				total += val
				if (ii + 1) % freq == 0:
					summ = session.run(model['nlls'], feed_dict = feeddict)
					summary.add_summary(summ, model['gsd'].eval())
					print datetime.datetime.now(), 'epoch', i, 'batch', ii, 'loss:', val, total
			else:
				vals, exps = session.run(model['p'], feed_dict = feeddict), np.argmax(feeddict[model['o']], 1)
				for exp, val, qid in zip(exps, vals, qidlist):
					if exp == val: total += 1
					evallist.append([qid, exp, val])
	return total if train else total, evallist

def ndcg(evallist, k = 20):
	evaldict, ndcglist = dict(), list()
	for qid, exp, val in evallist:
		if qid in evaldict: evaldict[qid].append([exp, val])
		else: evaldict[qid] = [[exp, val]]
	for qid in evaldict:
		dcg = sum([float(2 ** val - 1) / math.log((i + 2), 2) for i, exp, val in enumerate(sorted(evaldict[qid], key = lambda x: (x[1], x[0]), reverse = True)[: k])])
		idcg = sum([float(2 ** exp - 1) / math.log((i + 2), 2) for i, exp, val in enumerate(sorted(evaldict[qid], key = lambda x: x[0], reverse = True)[: k])])
		ndcglist.append(dcg / idcg)
	return sum(ndcglist) / len(ndcglist)

def handler(signum, frame):
	print datetime.datetime.now(), 'execution terminated'
	tf.train.Saver().save(sess, config.get('global', 'save'))
	sys.exit()

if __name__ == '__main__':
	signal.signal(signal.SIGINT, handler)
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])

	print datetime.datetime.now(), 'creating model'
	metamodel = config.get('global', 'model')
	model = globals()[metamodel].create(config[metamodel])

	with tf.Session() as sess:
		if sys.argv[2] == 'init':
			sess.run(tf.initialize_all_variables())
		else:
			tf.train.Saver().restore(sess, config.get('global', 'load'))
			summary = tf.train.SummaryWriter(config.get('global', 'logs'), sess.graph)

			if sys.argv[2] == 'train':
				print datetime.datetime.now(), 'training model'
				trainingloss = run(model, config, sess, summary, '%s/train' %config.get('global', 'data'), True)
				print datetime.datetime.now(), 'training loss', trainingloss
			else:
				print datetime.datetime.now(), 'testing model'
				testingaccuracy, evallist = run(model, config, sess, summary, '%s/%s' %(config.get('global', 'data'), sys.argv[2]), False)
				print datetime.datetime.now(), 'testing accuracy', testingaccuracy
				print datetime.datetime.now(), 'normalized discounted cumulative gain', ndcg(evallist)

		print datetime.datetime.now(), 'saving model'
		tf.train.Saver().save(sess, config.get('global', 'save'))
