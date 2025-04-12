import os
import glob
import torch
import numpy
import random
import soundfile
import cv2
import python_speech_features
import pickle
import shutil
import subprocess
import time
import pdb
import sys
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from talkNet import talkNet
from model.faceDetector.s3fd.box_utils import nms_
from model.faceDetector.s3fd import S3FD

def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	video_manager = VideoManager([args.videoFilePath])
	stats_manager = StatsManager()
	scene_manager = SceneManager(stats_manager)
	scene_manager.add_detector(ContentDetector(threshold = args.sceneDetectThresh))
	base_timecode = video_manager.get_base_timecode()
	video_manager.set_downscale_factor()
	video_manager.start()
	scene_manager.detect_scenes(frame_source = video_manager)
	scene_list = scene_manager.get_scene_list(base_timecode)
	savePath = os.path.join(args.pyworkPath,'scene.pckl')
	if scene_list == []:
		scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		pickle.dump(scene_list, fil)
	return scene_list

def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		dets.append([])
		for bbox in bboxes:
			dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, scene):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	facesList = pickle.load(open(os.path.join(args.pyworkPath,'faces.pckl'), "rb"))
	for shot in scene:
		if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
			framesList = list(range(shot[0].frame_num,shot[1].frame_num))
			if len(framesList) == 0: continue
			faces = [facesList[x] for x in framesList]
			shot_tracks = [] # Separate track for each shot
			while True:
				frameIndices = [] # List of frames containing the current track
				for fidx, f in enumerate(faces):
					for face in f:
						frameIndices.append((fidx, face))
				if len(frameIndices) == 0: # If no more faces, end the tracking
					break
				fidx, face = frameIndices[0]
				currentTrack = [face] # Track for face
				del faces[fidx][faces[fidx].index(face)]
				# Try to track the face
				for fidx in range(fidx + 1, len(faces)):
					# Check if the face is already in another track
					for face in faces[fidx]:
						if bb_intersection_over_union(currentTrack[-1]['bbox'], face['bbox']) > iouThres:
							currentTrack.append(face)
							del faces[fidx][faces[fidx].index(face)]
							continue
				if len(currentTrack) >= args.minTrack:
					frameNum    = numpy.array([ f['frame'] for f in currentTrack ])
					bboxes      = numpy.array([numpy.array(f['bbox']) for f in currentTrack])
					frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
					bboxesI     = []
					for ij in range(0,4):
						interpfn  = interp1d(frameNum, bboxes[:,ij])
						bboxesI.append(interpfn(frameI))
					bboxesI  = numpy.stack(bboxesI, axis=1)
					if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
						track = {'frame':frameI, 'bbox':bboxesI}
						shot_tracks.append(track)
			tracks.append(shot_tracks)
	savePath = os.path.join(args.pyworkPath,'tracks.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(tracks, fil)
	return tracks

def crop_video(args, tracks):
	# CPU: crop the face clips
	# First, we need to read all frames
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	vOut = {}
	for tidx, shot in enumerate(tracks):
		for fidx, track in enumerate(shot):
			cropPath = os.path.join(args.pycropPath, '%05d_%05d'%(tidx, fidx))
			os.makedirs(cropPath, exist_ok = True)
			vOut[cropPath] = []
			frame = track['frame']
			bbox = track['bbox']
			for bidx, bboxes in enumerate(bbox):
				fpath = flist[frame[bidx]]
				img = cv2.imread(fpath)
				H, W, _ = img.shape
				bboxes = numpy.round(bboxes).astype(numpy.int32)
				bboxes[0] = max(0, bboxes[0] - args.nPadding)
				bboxes[1] = max(0, bboxes[1] - args.nPadding)
				bboxes[2] = min(W, bboxes[2] + args.nPadding)
				bboxes[3] = min(H, bboxes[3] + args.nPadding)
				face = img[bboxes[1]:bboxes[3], bboxes[0]:bboxes[2]]
				cv2.imwrite(os.path.join(cropPath, '%05d.jpg'%(bidx)), face)
				vOut[cropPath].append(os.path.join(cropPath, '%05d.jpg'%(bidx)))
	return vOut

def extract_MFCC(args, vOut):
	# CPU: extract mfcc from audio file
	# Read audio file
	sample_rate, audio = wavfile.read(os.path.join(args.pyworkPath, 'audio.wav'))
	audio = numpy.float64(audio) / 32768.0 # Convert int16 to float
	for vidx, value in enumerate(vOut.items()):
		cropPath, _ = value
		faces = glob.glob(os.path.join(cropPath, '*.jpg'))
		faces.sort()
		fps = float(len(faces)) / float(len(audio) / sample_rate)
		# Extract MFCC
		mfcc = zip(*python_speech_features.mfcc(audio, sample_rate, numcep = 13, winlen = 0.025 * sample_rate / 16000, winstep = 0.010 * sample_rate / 16000))
		mfcc = numpy.stack([numpy.array(i) for i in mfcc])
		# Create video to audio mapping
		idx = numpy.linspace(0, len(mfcc) - 1, int(len(faces) * 100))
		mfcc = numpy.array([mfcc[int(i)] for i in idx])
		savePath = os.path.join(cropPath, 'mfcc.pkl')
		with open(savePath, 'wb') as fil:
			pickle.dump(mfcc, fil)
	return

def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained TalkNet
	s = talkNet()
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	# Evaluate the active speaker detection performance for each crop
	for idx, video in enumerate(files):
		faces = glob.glob(os.path.join(video, '*.jpg'))
		faces.sort()
		if len(faces) == 0: continue
		with open(os.path.join(video, 'mfcc.pkl'), 'rb') as fil:
			mfccs = pickle.load(fil)
		if len(mfccs) == 0: continue
		# Compute the scores for the faces
		faces_tensor = []
		for face in faces:
			faces_tensor.append(cv2.imread(face))
		faces_tensor = numpy.array(faces_tensor)
		mfccs_tensor = numpy.array(mfccs)
		scores = s.evaluate(faces_tensor, mfccs_tensor)
		allScores.append(scores)
	savePath = os.path.join(args.pyworkPath, 'scores.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(allScores, fil)
	return allScores

def visualization(tracks, scores, args):
	# CPU: visulize the result as the demo video
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	faces = pickle.load(open(os.path.join(args.pyworkPath,'faces.pckl'), "rb"))
	allTracks = []
	for shot in tracks:
		for track in shot:
			allTracks.append(track)
	# Visualization
	colorDict = {True:(0,255,0), False:(0,0,255)}
	vidTracks = []
	for tidx, track in enumerate(allTracks):
		if len(scores) <= tidx: continue
		frames = track['frame']
		bboxes = track['bbox']
		score = scores[tidx]
		for fidx, frame in enumerate(frames):
			vidTracks.append({'track':tidx, 'frame':frame, 'bbox':bboxes[fidx], 'score':score[fidx], 'active':score[fidx] > args.activeThresh})
	vidTracks.sort(key=lambda x: x['frame'])
	vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_out.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (int(args.videoWidth), int(args.videoHeight)))
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		for track in vidTracks:
			if track['frame'] == fidx:
				bbox = track['bbox']
				active = track['active']
				cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colorDict[active], 2)
				cv2.putText(image,'%s: %.3f'%(active, track['score']), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorDict[active], 2)
		vOut.write(image)
	vOut.release()
	return

def evaluate_col_ASD(args):
	# CPU: evaluate the performance of the active speaker detection on Columbia dataset
	# Download the dataset
	if os.path.exists(os.path.join(args.colSavePath, 'col_front.pkl')) == False:
		if os.path.exists(args.colSavePath) == False:
			os.makedirs(args.colSavePath)
		downloadCmd = 'wget http://www.ee.columbia.edu/~dpwe/sounds/aed/aed1.13.tgz -P %s'%(args.colSavePath)
		subprocess.call(downloadCmd, shell=True)
		unzipCmd = 'tar -xf %s/aed1.13.tgz -C %s'%(args.colSavePath, args.colSavePath)
		subprocess.call(unzipCmd, shell=True)
		subprocess.call('rm %s/aed1.13.tgz'%(args.colSavePath), shell=True)
		subprocess.call('mv %s/aed1.13/aed1/wavs/* %s'%(args.colSavePath, args.colSavePath), shell=True)
		subprocess.call('rm -r %s/aed1.13'%(args.colSavePath), shell=True)
		downloadCmd = 'wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/columbia_data.pkl -P %s'%(args.colSavePath)
		subprocess.call(downloadCmd, shell=True)
	# Load the dataset
	with open(os.path.join(args.colSavePath, 'columbia_data.pkl'), 'rb') as fil:
		colData = pickle.load(fil, encoding='latin1')
	with open(os.path.join(args.colSavePath, 'col_front.pkl'), 'wb') as fil:
		pickle.dump(colData, fil)
	# Evaluate the performance
	s = talkNet()
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	videoLabelDict = {'1': 'bell', '2': 'boll', '3': 'lieb', '4': 'sick', '5': 'long'}
	videoAppearedDict = {'bell': 0, 'boll': 0, 'lieb': 0, 'sick': 0, 'long': 0}
	videoMatchDict = {'bell': 0, 'boll': 0, 'lieb': 0, 'sick': 0, 'long': 0}
	for videoName in colData.keys():
		videoLabels = colData[videoName]['labels']
		videoFrames = colData[videoName]['video']
		videoAudio = colData[videoName]['audio']
		videoAudio = numpy.array(videoAudio)
		videoFrames = numpy.array(videoFrames)
		for fidx in range(len(videoLabels)):
			videoLabel = videoLabels[fidx]
			videoFrame = videoFrames[fidx]
			videoDict = videoLabelDict[videoName[0]]
			videoAppearedDict[videoDict] += 1
			# Compute the scores for the faces
			faces_tensor = numpy.array([videoFrame])
			mfccs_tensor = numpy.array([videoAudio[fidx]])
			scores = s.evaluate(faces_tensor, mfccs_tensor)
			allScores.append(scores)
			pred = scores[0] > args.activeColThresh
			if pred == videoLabel:
				videoMatchDict[videoDict] += 1
	# Compute the accuracy
	for videoDict in videoAppearedDict.keys():
		print('%s\t total %d, match %d, accuracy: %.2f'%(videoDict, videoAppearedDict[videoDict], videoMatchDict[videoDict], float(videoMatchDict[videoDict]) / float(videoAppearedDict[videoDict]) * 100))
	# Compute the F1 score
	for videoDict in videoAppearedDict.keys():
		videoKey = list(videoLabelDict.keys())[list(videoLabelDict.values()).index(videoDict)]
		labels = numpy.array([colData[videoKey + '_%d'%(i)]['labels'] for i in range(1, 10)]).flatten()
		scores = numpy.array(allScores)
		scores = scores.reshape(scores.shape[0])
		scores = scores[:len(labels)]
		predicts = scores > args.activeColThresh
		TP = sum(predicts * labels)
		TN = sum((1 - predicts) * (1 - labels))
		FP = sum(predicts * (1 - labels))
		FN = sum((1 - predicts) * labels)
		F1 = 2 * TP / (2 * TP + FP + FN)
		print('%s\t TP %d, TN %d, FP %d, FN %d, F1: %.1f'%(videoDict, TP, TN, FP, FN, F1 * 100))
	return

def main():
	# Training settings
	import argparse
	parser = argparse.ArgumentParser(description = 'The demo of TalkNet, e.g., python demoTalkNet.py --videoName xxx.mp4')
	parser.add_argument('--videoName',             type=str, default="",   help='Input video name')
	parser.add_argument('--videoPath',             type=str, default="demo/",   help='Path for input video')
	parser.add_argument('--savePath',              type=str, default="demo/",   help='Path for saving result')
	parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')
	parser.add_argument('--nDataLoaderThread',     type=int, default=10,   help='Number of workers for dataloader')
	parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
	parser.add_argument('--minTrack',              type=int, default=10,   help='Number of min frames for each shot')
	parser.add_argument('--numFailedDet',          type=int, default=10,   help='Number of missed detections allowed before tracking is stopped')
	parser.add_argument('--minFaceSize',           type=int, default=1,    help='Minimum face size in pixels')
	parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')
	parser.add_argument('--nPadding',              type=int, default=50,   help='Padding boundary of faces')
	parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columbia dataset')
	parser.add_argument('--colSavePath',           type=str, default="data/col",   help='Path for saving Columbia dataset')
	parser.add_argument('--sceneDetectThresh',     type=float, default=30, help='Threshold for scene detection')
	parser.add_argument('--activeThresh',          type=float, default=0.5, help='Threshold for active speaker detection')
	parser.add_argument('--activeColThresh',       type=float, default=0.55, help='Threshold for active speaker detection on Columbia dataset')
	args = parser.parse_args()
	if args.evalCol == True:
		if os.path.exists(args.pretrainModel) == False:
			downloadCmd = 'wget www.robots.ox.ac.uk/~vgg/software/lipsync/data/pretrain_TalkSet.model -P ./'
			subprocess.call(downloadCmd, shell=True)
		evaluate_col_ASD(args)
		quit()
	if args.videoName == "":
		raise ValueError("videoName should not be empty, please give the input video name")
	args.videoFilePath = os.path.join(args.videoPath, args.videoName + '.mp4')
	if os.path.exists(args.videoFilePath) == False:
		raise ValueError("Input video file does not exist, please check %s"%args.videoFilePath)
	# Initialization
	args.pyaviPath = os.path.join(args.savePath, args.videoName, 'pyavi')
	args.pyframesPath = os.path.join(args.savePath, args.videoName, 'pyframes')
	args.pyworkPath = os.path.join(args.savePath, args.videoName, 'pywork')
	args.pycropPath = os.path.join(args.savePath, args.videoName, 'pycrop')
	if os.path.exists(args.pyaviPath) == False:
		os.makedirs(args.pyaviPath)
	if os.path.exists(args.pyframesPath) == False:
		os.makedirs(args.pyframesPath)
	if os.path.exists(args.pyworkPath) == False:
		os.makedirs(args.pyworkPath)
	if os.path.exists(args.pycropPath) == False:
		os.makedirs(args.pycropPath)
	if os.path.exists(args.pretrainModel) == False:
		downloadCmd = 'wget www.robots.ox.ac.uk/~vgg/software/lipsync/data/pretrain_TalkSet.model -P ./'
		subprocess.call(downloadCmd, shell=True)
	# Extract video frames
	if os.path.exists(os.path.join(args.pyframesPath, '000001.jpg')) == False:
		command = ("ffmpeg -y -i %s -threads 1 -f image2 %s" % (args.videoFilePath, os.path.join(args.pyframesPath, '%06d.jpg')))
		output = subprocess.call(command, shell=True, stdout=None)
	if os.path.exists(os.path.join(args.pyworkPath, 'audio.wav')) == False:
		command = ("ffmpeg -y -i %s -threads 1 -f wav %s" % (args.videoFilePath, os.path.join(args.pyworkPath, 'audio.wav')))
		output = subprocess.call(command, shell=True, stdout=None)
	# Get video information
	videoCapture = cv2.VideoCapture(args.videoFilePath)
	if videoCapture.isOpened():
		args.videoWidth  = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
		args.videoHeight = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
		args.videoFrameRate = videoCapture.get(cv2.CAP_PROP_FPS)
		args.videoFrames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
		videoCapture.release()
	# Scene detection for the video frames
	scene = scene_detect(args)
	# Face detection for the video frames
	faces = inference_video(args)
	# Face tracking throughout the video
	allTracks = track_shot(args, scene)
	# Face clips cropping
	vOut = crop_video(args, allTracks)
	# MFCC extraction from audio
	extract_MFCC(args, vOut)
	# Active speaker detection by TalkNet
	allScores = evaluate_network(list(vOut.keys()), args)
	# Visualization
	visualization(allTracks, allScores, args)
	sys.stderr.write('%s has been processed, please check %s \r\n'%(args.videoFilePath, os.path.join(args.pyaviPath, 'video_out.avi')))

if __name__ == '__main__':
	main()
