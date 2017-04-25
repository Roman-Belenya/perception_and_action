import os
import sys
import re
import cPickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
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

def text_data(text_lines):
	name = re.search(r'Roman_\w+', text_lines[2])
	if name is None:
		name = re.search(r'Accuracy\w+', text_lines[2])
	time = re.search(r'\d+:\d+:\d+', text_lines[2])
	trial_len = re.search(r'\d+\.\d+', text_lines[4])

	return name.group(), time.group(), trial_len.group()

def columnise(data_lines):
	result = []
	for line in data_lines:
		result.append( map(float, line.split()) )
	return zip(*result)


def dictionarise(text_lines, data_lines):
	result = {}
	result['name'], result['time'], result['data_capture_period'] = text_data(text_lines)

	colheaders = text_lines[8][:-3].split('\t')
	colheaders[0].replace(' ', '_')
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
				result['short_trials']['t' + str(short_trial_no)]['fix'] = find_fixations(
																							result['short_trials']['t' + str(short_trial_no)]['averagexeye'],
																							result['short_trials']['t' + str(short_trial_no)]['averagezeye'])
				short_trial_no += 1
			else:
				result['trials']['t' + str(trial_no)] = dictionarise(text, data)
				result['trials']['t' + str(trial_no)]['fix'] = find_fixations(
																				result['trials']['t' + str(trial_no)]['averagexeye'],
																				result['trials']['t' + str(trial_no)]['averagezeye'])

				trial_no += 1
		print 'Done with:   {}'.format(f.name[:-4])
	return result


def add_participant(p_id, folder):

	if 'ExperimentData.pkl' in os.listdir(os.getcwd()):
		prior = load_data().keys()
		print prior
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


def find_fixations(eyex, eyez, dispersion_th = 0.01, duration_th = 0.1):

	eyex = list(eyex)
	eyez = list(eyez)

	result = { 'start_frame':[], 'end_frame':[], 'duration':[], 'dispersion':[], 'centre_x':[], 'centre_z':[] }

	window = [0, int(duration_th * 130)] # duration th should be in seconds
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
			result['duration'].append((window[1] - window[0] + 1) / 130.)
			result['dispersion'].append(dispersion(eyex, eyez, window))
			result['centre_x'].append(np.mean( eyex[window[0]:window[1]] ))
			result['centre_z'].append(np.mean( eyez[window[0]:window[1]] ))

			index += 1
			window = [window[1] + 1, window[1] + 13]

		else:
			window = [ x + 1 for x in window ]

	return result


def dic2mat():
	from scipy.io import savemat
	dic = load_data()
	savemat('ExperimentData.mat', dic)


def check_accuracy(trial):

	eye_x = np.array(trial['averagexeye']) * 100
	eye_z = np.array(trial['averagezeye']) * 100

	centre_x = 0.605 * 100
	centre_z = 0.33468 * 100

	error_x = np.array(trial['errorx']) * 100
	error_z = np.array(trial['errorz']) * 100
	error_y = np.array(trial['errory']) * 100

	dist = np.array(trial['totalerror']) * 100

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
	ax2.axhline(np.mean(dist), color = 'r', linestyle = ':', label = 'Mean')
	ax2.axhline(np.median(dist), color = '#FF8D0D', linestyle = ':', label = 'Median')
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


def get_accuracy_summary():

	d = load_data()

	fig = plt.figure()

	p_no = 1
	p_n = len(d.keys())

	for p in d.keys():
		for t in d[p]['accuracy'].keys():

			error = np.array(d[p]['accuracy'][t]['totalerror']) * 100
			ax = fig.add_subplot(p_n, 3, p_no)
			p_no += 1
			ax.plot(error, 'b-')
			ax.axhline(1, color = 'r')
			ax.set_ylim(0, 5)
			ax.set_ylabel(p + ' ' + t)
			if np.mean(error) < 1:
				ax.plot(0, 4.5, 'g.', markersize = 15)
			else:
				ax.plot(0, 4.5, 'r.', markersize = 15)
	plt.show()
