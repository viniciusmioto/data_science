#! /usr/bin/env python
# Word Embedding

import gzip
import gensim
import logging
import pickle
import csv
import numpy as np
import argparse

logging.basicConfig(level= logging.DEBUG)

VectorSize = 300
WindowSize = 15
MinCount = 2
DataTR = "reviews_data.txt.gz"
#DataTR = "imdb-tr.csv.gz"


def read_args():
	parser = argparse.ArgumentParser(description='Os parametros sao:')
	parser.add_argument('-m', '--method', type=str, required=True,
											help='1 - Media\n, 2 - Mediana')
	return parser.parse_args()

def read_input(input_file):
	"""This method reads the input file which is in gzip format"""

	logging.info("reading file {0}...this may take a while".format(input_file))

	with gzip.open (input_file, 'rb') as f:
		for i, line in enumerate (f):

			if (i%10000==0):
				logging.info ("read {0} reviews".format (i))
			yield gensim.utils.simple_preprocess (line)

def read_and_train():
	documents = list (read_input (DataTR))
	#documents =le list (read_input (DataTR))
	print ("Done reading data file")
	model = gensim.models.Word2Vec (documents, vector_size=VectorSize, window=WindowSize, min_count=MinCount, workers=10)
	print ("Training model")
	model.train(documents,total_examples=len(documents),epochs=10)
	print ("Model trained")
	return model

def save_model(model):
	print ('Saving Model into modelogensin file...')
	fileObject = open("modelogensim", 'wb')
	pickle.dump(model, fileObject)
	fileObject.close()

def open_model():
	fileObject = open("modelogensim", 'rb')
	model = pickle.load(fileObject)
	fileObject.close()
	return model

def read_csv():
	labels = {"neg":0, "pos":1}
	reviews = []
	#with open('imdb_master.csv') as csvfile:
	with open('imdb_master.csv', encoding="ISO-8859-1") as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if not(row['label']=="unsup"):
				reviews.append([(row['review'], labels[row['label']])])

	for j in range(0, len(reviews)):
		w = (gensim.utils.simple_preprocess(reviews[j][0][0]))
		w1 = [w, reviews[j][0][1]]
		reviews[j][0] = w1
	return reviews

## 1 Media
## 2 Mediana
def extract_features_mean(model, reviews, metric):
	result = []

	print ('Building Representation...')
	for j in range(0, len(reviews)):
		w =  reviews[j][0][0]
		
		v_mean = []
		
		for k in range (0, len(w)):
			
			try:
				v_mean.append(model.wv[w[k]])
			except KeyError:
				v_mean.append(np.zeros(VectorSize))
		

		if( metric == 1):
			result.append(np.mean(v_mean, axis=0))
		else:
			result.append(np.median(v_mean, axis=0))
		
		
	return result

#def extract_features_median(model, reviews):
#	result = []
#	print ('Building Representation...')
#	for j in range(0, len(reviews)):
#		w =  reviews[j][0][0]
#		print(v_mean)
#		for k in range (0, len(w)):
#			if w[k] in model.wv.vocab:
#				v_mean.append(model[w[k]])
#			else:
#				v_mean.append(np.zeros(VectorSize))
#		result.append(np.median(v_mean, axis=0))
#	return result


def write_results(result, reviews):
	print ('Saving train.txt file...')
	f = open("train.txt", "w")
	#line1 = str(len(reviews)/2) + " 150" + "\n"
	#f.write(line1)
	t = len(result)
	for i in range(0, int(t/2)):
		line = str(reviews[i][0][1]) + " "
		for j in range(0, len(result[i])):
			line = line + str(j) + ":" + str(result[i][j]) + " "
		line = line + " "  +  "\n"
		f.write(line)
	f.close()


	print ('Saving train.txt file...')
	f = open("test.txt", "w")
	#line1 = str(len(reviews)/2) + " 150" + "\n"
	#f.write(line1)
	for i in range(int(t/2), t):
		line = str(reviews[i][0][1]) + " "
		for j in range(0, len(result[i])):
			line = line + str(j) + ":" + str(result[i][j]) + " "
		line = line + " "  + "\n"
		f.write(line)
	f.close()


if __name__ == "__main__":
	args = read_args()
	m = int(args.method)
	#uncoment this to train word2vec
	model = read_and_train()
	#save_model(model)
	#model = open_model()
	reviews = read_csv()
	results = extract_features_mean(model, reviews, m)
	write_results(results, reviews)
	
