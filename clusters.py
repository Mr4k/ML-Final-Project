from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np
import json
import gzip

word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

parsed = parse("reviews_Electronics_5.json.gz")

numReviewsToViewInEachCategory = 2
wordsSeen = {}
wordVecArray = None
indicesToWord = []

catagories = {1:0, 2:0, 3:0, 4:0, 5:0}

unknownWords = 0

#this step collects all the words we will be using to keep our clustering problem to a minimum size
print "Preprocess Step"
for p in parsed:
	cat = p['overall']
	if catagories[cat] < numReviewsToViewInEachCategory:
		catagories[cat] += 1
	else:
		continue
	pStr = p['reviewText']
	delims = ',;.!?\n"'
	for delim in delims:
		pStr = pStr.replace(delim, ' ')
	for word in pStr.split():
		if word.lower() in wordsSeen:
			continue
		wordsSeen[word.lower()] = 1
		if wordVecArray is None:
			if word.lower() in word_vectors:
				wordVecArray = np.matrix(word_vectors[word.lower()])
				indicesToWord.append(word.lower())
			else:
				unknownWords += 1
		else:
			if word.lower() in word_vectors:
				wordVecArray = np.concatenate([wordVecArray, np.matrix(word_vectors[word.lower()])], axis = 0)
				indicesToWord.append(word.lower())
			else:
				unknownWords += 1
	done = True
	for cat in catagories:
		if catagories[cat] < numReviewsToViewInEachCategory:
			done = False
	if done:
		break

print wordVecArray.shape
print unknownWords

print "Begin Clustering"
numClusters = wordVecArray.shape[0]/15
clusters = []
for i in xrange(numClusters):
	clusters.append([i])
kmeansCluster = KMeans(n_clusters = numClusters).fit(wordVecArray)
print kmeansCluster.labels_
for i in xrange(len(kmeansCluster.labels_)):
	clusters[kmeansCluster.labels_[i]].append(indicesToWord[i])
for i in xrange(numClusters):
	print "\n" * 4
	print "CLUSTER ("+str(i)+")"
	print clusters[i]
	print "\n" * 4