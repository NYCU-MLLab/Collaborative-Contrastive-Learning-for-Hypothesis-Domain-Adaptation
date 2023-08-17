'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch, csv
from scipy import signal

class data_loader(object):
	def __init__(self, aug, train_list, train_path, dataset, musan_path, rir_path, num_frames, **kwargs):
		self.num_frames = num_frames
		# Load and configure augmentation files
		self.aug = aug
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-3] not in self.noiselist:
				self.noiselist[file.split('/')[-3]] = []
			self.noiselist[file.split('/')[-3]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

        # Load data & labels
		self.data_list  = []
		self.data_label = []

		if dataset == 'commonvoice':
			with open(train_list, newline='') as csvfile:
				lines = csv.reader(csvfile, delimiter='	')
				for line in lines:
					speaker_label 	= int(line[0])
					file_name		= os.path.join(train_path, line[1])
					self.data_label.append(speaker_label)
					self.data_list.append(file_name)
		elif dataset == 'cn1':
			path = os.path.join(train_path,"**", "*.flac")
			for f in glob.glob(path, recursive=True):
				try:
					speaker_label = f.split(os.path.sep)[-2].replace('id','')
					# avoid test speakers for train
					if int(speaker_label) >= 800:
						continue
					else:
						file_name     = f
						self.data_label.append(int(speaker_label))
						self.data_list.append(file_name)
				except ValueError:
					logger.info(f"Malformed path: {f}")
		elif dataset == 'cn2':
			lines = open(train_list).read().splitlines()
			for index, line in enumerate(lines):
				path = os.path.join(train_path, line.split()[0], "*.flac")
				for f in glob.glob(path, recursive=True):
					#speaker_label = f.split(os.path.sep)[-2].replace('id','')
					#speaker_label = index + 1211
					speaker_label = index
					file_name     = f
					self.data_label.append(int(speaker_label))
					self.data_list.append(file_name)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = soundfile.read(self.data_list[index])
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)
		# Data Augmentation
		if self.aug == False:   # Original
			audio = audio
			return torch.FloatTensor(audio[0]), self.data_label[index]
		else:
			augtype = random.randint(1,5)
			ori_audio = audio
			if augtype == 1: # Reverberation
				audio = self.add_rev(audio)
			elif augtype == 2: # Babble
				audio = self.add_noise(audio, 'speech')
			elif augtype == 3: # Music
				audio = self.add_noise(audio, 'music')
			elif augtype == 4: # Noise
				audio = self.add_noise(audio, 'noise')
			elif augtype == 5: # Television noise
				audio = self.add_noise(audio, 'speech')
				audio = self.add_noise(audio, 'music')
			return torch.FloatTensor(audio[0]), torch.FloatTensor(ori_audio[0]),   self.data_label[index]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4)
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio