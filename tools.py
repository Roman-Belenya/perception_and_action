import os
import sys
import re
import cPickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style, patches, lines
from scipy.io import savemat
style.use('ggplot')


def load_data():
	obj = {}
	with open('ExperimentData.pkl', 'rb') as f:
		while True:
			try:
				obj.update(cPickle.load(f))
			except EOFError:
				break
	return obj

def save_data(object_name):
	with open('ExperimentData.pkl', 'ab') as f:
		cPickle.dump(object_name, f, cPickle.HIGHEST_PROTOCOL)


def manual_save_data(object_name, data_file):
	ans = raw_input('The data will be overwritten. Continue? Yes/no')
	if ans == 'Yes':
		with open(data_file, 'wb') as f:
			cPickle.dump(object_name, f, cPickle.HIGHEST_PROTOCOL)
	else:
		print 'Cancelling ...'
		return

def manual_load_data(data_file):
	with open(data_file, 'rb') as f:
		data = cPickle.load(f)
	return data



def text_data(text_lines):
	measure_rate = re.search(r'\d+.\d+', text_lines[3]).group()
	measure_rate = float(measure_rate)
	name = re.search(r'Roman_\w+', text_lines[2])
	if name is None:
		name = re.search(r'Accuracy\w+', text_lines[2])
	time = re.search(r'\d+:\d+:\d+', text_lines[2])
	date = re.search(r'\d+-\d+-\d+', text_lines[2])
	trial_len = re.search(r'\d+\.\d+', text_lines[4])

	return name.group(), time.group(), date.group(), trial_len.group(), measure_rate

def columnise(data_lines):
	result = []
	for line in data_lines:
		result.append( map(float, line.split()) )
	return zip(*result)


def dictionarise(text_lines, data_lines):
	result = {}
	result['name'], result['time'], result['date'], result['data_capture_period'], result['measurement_rate'] = text_data(text_lines)

	colheaders = text_lines[8][:-3].split('\t')
	colheaders[0] = colheaders[0][:-2]
	# colheaders[0].replace(' ', '_')

	columns = columnise(data_lines)

	for colheader in colheaders:
		result[colheader.lower()] = columns[colheaders.index(colheader)]

	return result


def organise(files_folder):

	files = [ file for file in os.listdir(files_folder) if file.endswith('.exp') ]
	result = {}
	result['trials'] = {}
	result['short_trials'] = {}
	result['accuracy'] = {}

	accuracy_trial_no = 1
	short_trial_no = 1
	trial_no = 1

	for file in files:
		with open(os.path.join(files_folder, file), 'rb') as f:
			lines = f.readlines()
			text = lines[:9]
			data = lines[9:]

			if 'Accuracy' in f.name:
				result['accuracy']['t' + str(accuracy_trial_no)] = dictionarise(text, data)
				accuracy_trial_no += 1
			elif 'SHORT' in f.name:
				result['short_trials']['t' + str(short_trial_no)] = dictionarise(text, data)
				result['short_trials']['t' + str(short_trial_no)]['fix'] = find_fixations(result['short_trials']['t' + str(short_trial_no)]['averagexeye'],
					result['short_trials']['t' + str(short_trial_no)]['averagezeye'])
				short_trial_no += 1
			else:
				result['trials']['t' + str(trial_no)] = dictionarise(text, data)
				result['trials']['t' + str(trial_no)]['fix'] = find_fixations(result['trials']['t' + str(trial_no)]['averagexeye'],
					result['trials']['t' + str(trial_no)]['averagezeye'])

				trial_no += 1
		print 'Done with:   {}'.format(f.name[:-4])
	return result


def add_participant(p_id, folder):
	''' Creates a new participant in the data file.

	Usage: add_participant(p_id = 'P00', folder = '../P00')'''

	if 'ExperimentData.pkl' in os.listdir(os.getcwd()):
		prior = load_data().keys()
		if p_id in prior:
			x = raw_input('Participant {} already exists. Overwrite? Yes/no\n'.format(p_id))
			while x != 'Yes' and x != 'no':
				x = raw_input('Bad input. Overwrite participant {}? Yes/no\n'.format(p_id))
			if x == 'no':
				print 'Na net i suda net'
				return
			else:
				print 'Overwriting participant {}'.format(p_id)
	else:
		x = raw_input('The ExperimentData.pkl file is not found. Create? Yes/no\n')
		while x != 'Yes' and x != 'no':
			x = raw_input('Bad input. Create new data file? Yes/no\n')
		if x == 'no':
			print 'Cancelling...'
			return
		else:
			with open('ExperimentData.pkl', 'wb'):
				print 'New experiment data file created in {}'.format(os.getcwd())

	current = {}
	current[p_id] = organise(folder)
	print '-----------------------------------------------------------'
	print 'Participant\'s id:               {}'.format(p_id)
	print 'Measurement rate:                {}'.format(current[p_id]['trials']['t1']['measurement_rate'])
	print '---'
	print 'Number of accuracy trials:       {}'.format(len( current[p_id]['accuracy'].keys() ))
	print 'Number of short trials:          {}'.format(len( current[p_id]['short_trials'].keys() ))
	print 'Number of experimental trials:   {}\n'.format(len( current[p_id]['trials'].keys() ))

	x = raw_input('Save data? Yes/no\n')
	if x == 'Yes':
		save_data(current)
		print('\nAll done!! :)')
	else:
		print 'Cancelling...\n'
		return


def dispersion(eyex, eyez, window):
	d = ( max(eyex[window[0]:window[1]]) - min(eyex[window[0]:window[1]]) ) + \
		( max(eyez[window[0]:window[1]]) - min(eyez[window[0]:window[1]]) )
	return d


def find_fixations(eyex, eyez, dispersion_th = 0.01, duration_th = 0.1, measure_rate = 130.):

	eyex = list(eyex)
	eyez = list(eyez)

	result = { 'start_frame':[], 'end_frame':[], 'duration':[], 'dispersion':[], 'centre_x':[], 'centre_z':[] }

	window = [0, int(duration_th * measure_rate)] # duration th should be in seconds
	index = 1

	while window[1] < len(eyex):
		d = dispersion(eyex, eyez, window)
		if d <= dispersion_th:

			while d <= dispersion_th and window[1] < len(eyex):
				window[1] += 1
				d = dispersion(eyex, eyez, window)

			if window[1] != len(eyex):
				window[1] -= 1

			result['start_frame'].append(window[0])
			result['end_frame'].append(window[1])
			result['duration'].append((window[1] - window[0] + 1) / measure_rate)
			result['dispersion'].append(dispersion(eyex, eyez, window))
			result['centre_x'].append(np.mean( eyex[window[0]:window[1]] ))
			result['centre_z'].append(np.mean( eyez[window[0]:window[1]] ))

			index += 1
			window = [window[1] + 1, window[1] + int(measure_rate * duration_th)]

		else:
			window = [ x + 1 for x in window ]

	return result


def dic2mat():
	dic = load_data()
	savemat('ExperimentData.mat', dic)


def check_accuracy(data, participant, trial = 't1'):
	''' Make an accuracy graph.

	Usage: check_accuracy(data = d, participant = 'P00', trial = 't0')'''

	trial = data[participant]['accuracy'][trial]

	eye_x = np.array(trial['averagexeye']) * 100
	eye_z = np.array(trial['averagezeye']) * 100

	centre_x = 0.605 * 100
	centre_z = 0.33468 * 100

	error_x = np.array(trial['errorx']) * 100
	error_z = np.array(trial['errorz']) * 100
	error_y = np.array(trial['errory']) * 100

	dist = np.array(trial['totalerror']) * 100
	mean_dist = np.mean(dist)
	median_dist = np.median(dist)

	r1 = 1
	r2 = 0.5
	theta = np.arange(0, 2.01 * np.pi, np.pi / 100)
	x = r1 * np.cos(theta) + centre_x
	z = r1 * np.sin(theta) + centre_z

	fig = plt.figure()
	fig.subplots_adjust(wspace=0.05, top=1, right=0.97, left=0.03, bottom=0)


	ax1 = fig.add_subplot(131, aspect = 'equal')
	ax1.plot(eye_x, eye_z, 'b.', alpha = 0.6)
	ax1.plot(x, z, 'r-')
	ax1.plot(r2 * np.cos(theta) + centre_x, r2 * np.sin(theta) + centre_z, ':', color = '#FF8D0D')
	ax1.plot(np.mean(eye_x), np.mean(eye_z), 'r.', markersize = 15)
	ax1.plot(np.median(eye_x), np.median(eye_z), '.', color = '#FF8D0D', markersize = 15)
	ax1.plot(centre_x, centre_z, 'k+', markersize = 10)
	ax1.set_xlim([centre_x - 2, centre_x + 2])
	ax1.set_ylim([centre_z - 2, centre_z + 2])


	ax2 = fig.add_subplot(132)
	ax2.plot(dist, 'b-')
	ax2.axhline(1, color = 'r')
	ax2.axhline(mean_dist, color = 'r', linestyle = ':', label = 'Mean')
	ax2.axhline(median_dist, color = '#FF8D0D', linestyle = ':', label = 'Median')
	ax2.set_ylim([0, 5])
	x0, x1 = ax2.get_xlim()
	y0, y1 = ax2.get_ylim()
	asp = (x1 - x0) / (y1 - y0)
	ax2.set_aspect(asp)
	ax2.legend()

	ax3 = fig.add_subplot(133)
	ax3.plot(error_x, 'b-', label = 'Error X')
	ax3.plot(error_z, 'g-', label = 'Error Z')
	ax3.plot(error_y, 'y-', label = 'Error Y')
	ax3.axhline(1, color = 'r')
	ax3.axhline(-1, color = 'r')
	ax3.axhline(0, color = 'k', linestyle = ':')
	ax3.set_ylim([-2.5, 2.5])
	x0, x1 = ax3.get_xlim()
	y0, y1 = ax3.get_ylim()
	asp = (x1 - x0) / (y1 - y0)
	ax3.set_aspect(asp)
	ax3.legend()

	plt.show()
	return(mean_dist, median_dist)


def get_accuracy_summary(data):

	d = data

	fig = plt.figure()

	p_no = 1
	p_n = len(d.keys())

	for p in d.keys():
		ax = fig.add_subplot(np.ceil(p_n / 3.), 3, p_no)
		for t in d[p]['accuracy'].keys():
			error = np.array(d[p]['accuracy'][t]['totalerror']) * 100
			ax.plot(error, label = t)
			ax.axhline(1, color= 'r')
			ax.set_ylim(0, 5)
			ax.set_title(p)
		plt.legend()
		p_no += 1
	plt.show()


def choose_marker(data, participant, marker = 'index'):
	''' Creates an overlayed graph of all reaches for the specified marker

	Usage: choose_marker(data = d, participant = 'P00', marker = 'index') '''

	if participant not in data.keys():
		raise NameError('Participant {} does notexist'.format(participant))

	if marker == 'index':
		mark = ('index7x', 'index8x', 'index7y', 'index8y', 'index7z', 'index8z')
		titles = ('Index7', 'Index8')
	elif marker == 'thumb':
		mark = ('thumb9x', 'thumb10x', 'thumb9y', 'thumb10y', 'thumb9z', 'thumb10z')
		titles = ('Thumb9', 'Thumb10')
	elif marker == 'wrist':
		mark = ('wrist11x', 'wrist12x', 'wrist11y', 'wrist12y', 'wrist11z', 'wrist12z')
		titles = ('Wrist11', 'Wrist12')
	else:
		raise NameError('Marker {} does not exist. Specify index, thumb or wrist'.format(marker))

	d = data
	p = participant

	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222, sharex = ax1)
	ax3 = fig.add_subplot(223, sharex = ax1)
	ax4 = fig.add_subplot(224, sharex = ax1)

	for trial in d[p]['trials'].values():
		if 'LeftToRight' in trial['name']:
			ax1.plot(trial[mark[0]], 'b-', linewidth = 0.5, alpha = 0.5)
			ax2.plot(trial[mark[1]], 'b-', linewidth = 0.5, alpha = 0.5)

			ax1.plot(trial[mark[2]], 'r-', linewidth = 0.5, alpha = 0.5)
			ax2.plot(trial[mark[3]], 'r-', linewidth = 0.5, alpha = 0.5)

			ax1.plot(trial[mark[4]], 'g-', linewidth = 0.5, alpha = 0.5)
			ax2.plot(trial[mark[5]], 'g-', linewidth = 0.5, alpha = 0.5)

		elif 'RightToLeft' in trial['name']:
			ax3.plot(trial[mark[0]], 'b-', linewidth = 0.5, alpha = 0.5)
			ax4.plot(trial[mark[1]], 'b-', linewidth = 0.5, alpha = 0.5)

			ax3.plot(trial[mark[2]], 'r-', linewidth = 0.5, alpha = 0.5)
			ax4.plot(trial[mark[3]], 'r-', linewidth = 0.5, alpha = 0.5)

			ax3.plot(trial[mark[4]], 'g-', linewidth = 0.5, alpha = 0.5)
			ax4.plot(trial[mark[5]], 'g-', linewidth = 0.5, alpha = 0.5)


	ax1.set_title(titles[0])
	ax2.set_title(titles[1])
	ax1.set_ylabel('Rightward')
	ax3.set_ylabel('Leftward')
	plt.suptitle(p)
	plt.show()


def check_visible(data, participant, index = 'index8', thumb = 'thumb9'):
	'''Check accuracy on visible target trials only. Creates a scatterplot of grasps relative to the target. This is to check how well a participant performed the task.

	Usage: check_visible(data = d, partitipant = 'P00', index = 'index8', thumb = 'thumb9')'''

	d = data
	p = participant

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.add_patch(patches.Rectangle((-2, -2), 4, 4, color = [0.8, 0.8, 0.8]))

	dist_x = []

	for trial in d[p]['trials'].values():

		if 'Visible' in trial['name']:

			if index != None:
				dx_index = (trial[index + 'x'][-1] - trial['objectx'][-1]) * 100
				dz_index = (trial[index + 'z'][-1] - trial['objectz'][-1]) * 100
				ax.plot(dx_index, dz_index, 'r^')
				dist_x.append(abs(dx_index))

			if thumb != None:
				dx_thumb = (trial[thumb + 'x'][-1] - trial['objectx'][-1]) * 100
				dz_thumb = (trial[thumb + 'z'][-1] - trial['objectz'][-1]) * 100
				ax.plot(dx_thumb, dz_thumb, 'bv')

			if index != None and thumb != None:
				ax.add_line(lines.Line2D([dx_thumb, dx_index], [dz_thumb, dz_index], color = 'k', linewidth = 1, alpha = 0.1))

	xl, xr = -10, 10
	zb, zt = -10, 10
	ax.set_xlim(xl, xr)
	ax.set_ylim(zb, zt)
	asp = (xr - xl) / (zt - zb)
	ax.set_aspect(asp)
	ax.set_title(p)
	plt.show()

	if dist_x != []:
		print 'Mean   = {} +- {}\nMedian = {}\n'.format(np.mean(dist_x), np.std(dist_x), np.median(dist_x))
		return np.mean(dist_x), np.std(dist_x)



def draw_cues(axis, ybottom = None, ytop = None):
	if ybottom is None:
		ybottom = axis.get_ylim()[0]
	if ytop is None:
		ytop = axis.get_ylim()[1]
	cues = [0.353 + n * 0.072 for n in range(8)]

	for cue in cues:
		axis.add_line(lines.Line2D([cue, cue], [ybottom, ytop], color = 'k', alpha = 0.5))



def get_intercept(p0, p1, q):
    '''Make a stright line between p0 and p1, find the perpendicular vector to this line passing through q, determine the intercept and distance.

    Usage: intercept, distance, error = get_intercept([p0x, p0y, p0z], [p1x, p1y, p1z], [qx, qy, qz])

	intercept is the point on the straight line where it crosses the perpendicular vector from point q
	distance is the Eucledian distance between q and the intercept
	error is the computational inaccuracy from computing the dot product between the straight and the perpendicular line. Should be very close to 0.

    Reference:
    https://www.youtube.com/watch?v=0lG53-ogF2k'''

    p0 = np.array(p0); p1 = np.array(p1); q = np.array(q)

    v = np.array(p1 - p0) # direction of the sraight line
    pq = np.array(p0 - q) # point on the direction vector for the given point

    t = -np.sum(v * pq) / np.sum(v**2) # this is the dot product between straight and perp line rearranged for t.
    intercept = p0 + t * v

    distance = np.sqrt( np.sum((q - intercept)**2) )

    error = np.dot(v, intercept - q)

    return intercept, distance, error
