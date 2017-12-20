##************************************************************************************************
##Libraries
##************************************************************************************************
import os
import numpy as np
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
import scipy
import collections
##************************************************************************************************



##************************************************************************************************
## Path of stemmed output documents
##************************************************************************************************
path = os.getcwd()+'/stemmed_output/'
##************************************************************************************************



##************************************************************************************************
## Reading all documents
##************************************************************************************************
documents_final = []
for filename in os.listdir(path):
	file1 = open(path+filename,'r')
	text1 = file1.readlines()
	#print(text1)
	documents_final.append(str(text1))

documents = tuple(documents_final)
##************************************************************************************************



##************************************************************************************************
## Converting into tf-idf matrix 
##************************************************************************************************
tfidf_vectorizer = TfidfVectorizer()
tf_arrrix = tfidf_vectorizer.fit_transform(documents)
tf_arr = tf_arrrix.toarray()
##************************************************************************************************



##************************************************************************************************
## Converting documents into boolean vectors
##************************************************************************************************
bool_arr = []
for i in tf_arr:
	bool_arr_t = []
	for j in i:
		if j > 0:
			bool_arr_t.append(1)
		else:
			bool_arr_t.append(0)
	bool_arr.append(bool_arr_t)
##************************************************************************************************



##************************************************************************************************
## term list of all documents
##************************************************************************************************
tlist = [] 
for x in documents_final:
	y = str(x).split()
	tlist.append(y)
##************************************************************************************************
            


##************************************************************************************************
##1. Cosine: u.v/[u][v]
##************************************************************************************************
def cosine_similarity(tf_arr):
	cos_sim = []
	for i in tf_arr:
		cos_sim_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.cosine(i, j, w=None)
			cos_sim_t.append("{:2.4f}".format(1-cs))
		cos_sim.append(cos_sim_t)
	return cos_sim
##************************************************************************************************



##************************************************************************************************
##2. Braycurtis: |u(i)-v(i)|/|u(i)+v(i)|
##************************************************************************************************
def braycurtis(tf_arr):
	bray_curt = []
	for i in tf_arr:
		bray_curt_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.braycurtis(i, j, w=None)
			bray_curt_t.append("{:2.4f}".format(1-cs))
		bray_curt.append(bray_curt_t)
	return bray_curt
##************************************************************************************************



##************************************************************************************************
##3. Canberra: |u(i)-v(i)|/[u(i)]+[v(i)]
##************************************************************************************************
def canberra(tf_arr):
	can_berra = []
	for i in tf_arr:
		can_berra_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.canberra(i, j, w=None)
			if cs != 0:
				cs = 1/cs
			else:																			 
				cs = 1
			can_berra_t.append("{:2.4f}".format(cs))
		can_berra.append(can_berra_t)
	return can_berra
##************************************************************************************************



##************************************************************************************************
## 4. Chebyshev: max(|u(i)-v(i)|)
## ************************************************************************************************
def chebyshev(tf_arr):
	cheby_shev = []
	for i in tf_arr:
		cheby_shev_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.chebyshev(i, j)
			cheby_shev_t.append("{:2.4f}".format(1-cs))
		cheby_shev.append(cheby_shev_t)
	return cheby_shev
##************************************************************************************************



##************************************************************************************************
##5. Cityblock: sum(|u(i)-v(i)|)
##************************************************************************************************
def cityblock(tf_arr):
	city_block = []
	for i in tf_arr:
		city_block_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.cityblock(i, j, w=None)
			if cs != 0:
				cs = 1/cs
			else:																			 
				cs = 1
			city_block_t.append("{:2.4f}".format(cs))
		city_block.append(city_block_t)
	return city_block
##************************************************************************************************



##************************************************************************************************
##6.Co-relation:
##************************************************************************************************
def correlation(tf_arr):
	co_relation = []
	for i in tf_arr:
		co_relation_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.correlation(i, j, w=None, centered=True)
			co_relation_t.append("{:2.4f}".format(1-cs))
		co_relation.append(co_relation_t)
	return co_relation
##************************************************************************************************



##************************************************************************************************
##7.Euclidean: 
##************************************************************************************************
def euclidean(tf_arr):
	euclidean_ = []
	for i in tf_arr:
		euclidean_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.euclidean(i, j, w=None)
			if cs != 0:
				cs = cs/2
			else:																			 
				cs = 1
			euclidean_t.append("{:2.4f}".format(cs))
		euclidean_.append(euclidean_t)
	return euclidean_
##************************************************************************************************



##************************************************************************************************
##8.Minkowski:
##************************************************************************************************
def minkowski(tf_arr):
	minkowski_ = []
	for i in tf_arr:
		minkowski_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.minkowski(i, j, p=3, w=None)
			minkowski_t.append("{:2.4f}".format(1-cs))
		minkowski_.append(minkowski_t)
	return minkowski_
##************************************************************************************************



##************************************************************************************************
##9.Sqeuclidean:
##************************************************************************************************
def sqeuclidean(tf_arr):
	sq_euclidean = []
	for i in tf_arr:
		sq_euclidean_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.sqeuclidean(i, j, w=None)
			if cs != 0:
				cs = cs/2
			else:																			 
				cs = 1
			sq_euclidean_t.append("{:2.4f}".format(cs))
		sq_euclidean.append(sq_euclidean_t)
	return sq_euclidean
##************************************************************************************************



##************************************************************************************************
##10.Wminkowski:
##************************************************************************************************
def wminkowski(tf_arr):
	w_minkowski = []
	for i in tf_arr:
		w_minkowski_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.wminkowski(i, j, p=3, w=2)
			if cs != 0:
				cs = cs/2
			else:																			 
				cs = 1
			w_minkowski_t.append("{:2.4f}".format(cs))
		w_minkowski.append(w_minkowski_t)
	return w_minkowski
##************************************************************************************************



##************************************************************************************************
##11.Hamming:
##************************************************************************************************
def hamming(tf_arr):
	hamming_ = []
	for i in tf_arr:
		hamming_t = []
		for j in tf_arr:
			cs=scipy.spatial.distance.hamming(i, j, w=None)
			hamming_t.append("{:2.4f}".format(1-cs))
		hamming_.append(hamming_t)
	return hamming_
##************************************************************************************************



##************************************************************************************************
##12.jaccard
##************************************************************************************************
def jaccard(bool_arr):
	jac=[]
	for i in bool_arr:
		jac_t = []
		for j in bool_arr:
			cs=scipy.spatial.distance.jaccard(i,j)
			jac_t.append("{:2.4f}".format(1-cs))
		jac.append(jac_t)
	return jac
##************************************************************************************************



##************************************************************************************************
##13.dice
##************************************************************************************************
def dice(bool_arr):
	dice_=[]
	for i in bool_arr:
		dice_t = []
		for j in bool_arr:
			cs=scipy.spatial.distance.dice(i,j)
			dice_t.append("{:2.4f}".format(1-cs))
		dice_.append(dice_t)
	return dice_
##************************************************************************************************



##************************************************************************************************
##14.russellrao
##************************************************************************************************
def rusel(bool_arr1,bool_arr2):
    count = 0
    n = 0
    for i in range(len(bool_arr1)):
        if bool_arr1[i] == bool_arr2[i] and bool_arr1[i] == 1:
            count = count+1
            n += 1
        elif bool_arr1[i] == 1 or bool_arr2 == 1:
            n += 1
    return (n-count)/n

def russellrao_(bool_arr):
	rus=[]
	for i in bool_arr:
		rus_t = []
		for j in bool_arr:
			cs=rusel(i,j)
			rus_t.append("{:2.4f}".format(1-cs))
		rus.append(rus_t)
	return rus
##************************************************************************************************



##************************************************************************************************
##15.rogers-tanimoto
##************************************************************************************************
def rogertanimoto(bool_arr):
	rogers=[]
	for i in bool_arr:
		rogers_t=[]
		for j in bool_arr:
			cs=scipy.spatial.distance.rogerstanimoto(i,j)
			rogers_t.append("{:2.4f}".format(1-cs))
		rogers.append(rogers_t)
	return rogers
##************************************************************************************************



##************************************************************************************************
##16.sokalmichener
##************************************************************************************************
def sokalmichener(bool_arr):
	sokalmi=[]
	for i in bool_arr:
		sokalmi_t=[]
		for j in bool_arr:
			cs=scipy.spatial.distance.sokalmichener(i,j)
			sokalmi_t.append("{:2.4f}".format(1-cs))
		sokalmi.append(sokalmi_t)
	return sokalmi
##************************************************************************************************



##************************************************************************************************
##17.SokalSneath
##************************************************************************************************
def sokalsneath(bool_arr):
	sokalsn=[]
	for i in bool_arr:
		sokalsn_t=[]
		for j in bool_arr:
			cs=scipy.spatial.distance.sokalsneath(i,j)
			sokalsn_t.append("{:2.4f}".format(1-cs))
		sokalsn.append(sokalsn_t)
	return sokalsn
##************************************************************************************************



##************************************************************************************************
##18.yule
##************************************************************************************************
def yule(bool_arr):
	yule_=[]
	for i in bool_arr:
		yule_t=[]
		for j in bool_arr:
			cs=scipy.spatial.distance.yule(i,j)
			yule_t.append("{:2.4f}".format(1-cs))
		yule_.append(yule_t)
	return yule_
##************************************************************************************************



##************************************************************************************************
##Inversions
##************************************************************************************************
count = 0
def inversionsCount(x):
    global count
    midsection = int(len(x) / 2)
    #print(midsection)
    leftArray = x[:midsection]
    rightArray = x[midsection:]
    if len(x) > 1:
        # Divid and conquer with recursive calls
        # to left and right arrays similar to
        # merge sort algorithm
        inversionsCount(leftArray)
        inversionsCount(rightArray)
        
        # Merge sorted sub-arrays and keep
        # count of split inversions
        i, j = 0, 0
        a = leftArray; b = rightArray
        for k in range(len(a) + len(b) + 1):
            if a[i] <= b[j]:
                x[k] = a[i]
                count += (len(b) - j)
                i += 1
                if i == len(a) and j != len(b):
                    while j != len(b):
                        k +=1
                        x[k] = b[j]
                        j += 1
                    break
            elif a[i] > b[j]:
                x[k] = b[j]
                j += 1
                if j == len(b) and i != len(a):
                    while i != len(a):
                        k+= 1
                        x[k] = a[i]
                        i += 1                    
                    break   
    return x
##************************************************************************************************



##************************************************************************************************
##19. Structured based similarity
##************************************************************************************************
def structBasedSimilarity(tlist):
	global count
	ss = []
	for i in range(0,len(tlist)):
		ss_t = []
		tempDict = collections.OrderedDict.fromkeys(tlist[i])
		count1 = 0;
		for j in tempDict.keys():
			tempDict[j] = count1
			count1 = count1+1
		for j in range(0,len(tlist)):
			tempDict2 = collections.OrderedDict()
			for k in range(0,len(tlist[j])):
				if tlist[j][k] in tempDict and tlist[j][k] not in tempDict2:
					tempDict2[tlist[j][k]] = k
			finalList = []
			for k in tempDict2.keys():
				finalList.append(tempDict[k])
			inversionsCount(finalList)
			numOfInv = count
			count = 0
			if len(finalList) <= 1:
				ss_t.append(-1)
			else:
				ss_t.append("{:2.4f}".format((2*numOfInv)/(len(finalList)*(len(finalList)-1))))
		ss.append(ss_t)
	return ss
##************************************************************************************************



##************************************************************************************************
##similarity matrix
##************************************************************************************************
def similarityMatrix(tf_arr,bool_arr,tlist):
	#distance 
	m1 = cosine_similarity(tf_arr)
	m2 = braycurtis(tf_arr)
	m3 = canberra(tf_arr)
	m4 = chebyshev(tf_arr)
	m5 = cityblock(tf_arr)
	m6 = correlation(tf_arr)
	m7 = euclidean(tf_arr)
	m8 = minkowski(tf_arr)
	m9 = sqeuclidean(tf_arr)
	m10 = wminkowski(tf_arr)
	

	#boolean
	m11 = hamming(tf_arr) 
	m12 = jaccard(bool_arr)
	m13 = dice(bool_arr)
	m14 = russellrao_(bool_arr)
	m15 = rogertanimoto(bool_arr)
	m16 = sokalmichener(bool_arr)
	m17 = sokalsneath(bool_arr)
	m18 = yule(bool_arr)

	#structured 
	m19 = structBasedSimilarity(tlist)

	mat = []
	mat.append(m1)
	mat.append(m2)
	mat.append(m3)
	mat.append(m4)
	mat.append(m5)
	mat.append(m6)
	mat.append(m7)
	mat.append(m8)
	mat.append(m9)
	mat.append(m10)
	mat.append(m11)
	mat.append(m12)
	mat.append(m13)
	mat.append(m14)
	mat.append(m15)
	mat.append(m16)
	mat.append(m17)
	mat.append(m18)
	mat.append(m19)

	return mat
##************************************************************************************************



##************************************************************************************************
##similarity matrix
##************************************************************************************************
simMat = similarityMatrix(tf_arr,bool_arr,tlist)

##************************************************************************************************
##Average of all similarities
##************************************************************************************************
avgList = []

for i in tf_arr:
    avgL = []
    for j in tf_arr:
        avgL.append(0)
    avgList.append(avgL)

for i in range(0,len(simMat)):
    for j in range(0,len(simMat[i])):
            simMat[i][j] = list(map(float, simMat[i][j]))


for i in range(0,len(simMat)):
    for j in range(0,len(simMat[i])):
        for k in range(0,len(simMat[i][j])):
            avgList[j][k] = avgList[j][k] + simMat[i][j][k]
##************************************************************************************************



##************************************************************************************************
##Cohesion measure (Harmonic Mean)
##************************************************************************************************
def harmonic_mean(avgList):
    sum = 0
    for i in range(len(avgList)):
        if avgList[i] == 0:
            return 0
        sum += (1/avgList[i])
    if sum != 0:
        return 1/sum


for i in range(len(avgList)):
    for j in range(len(avgList[i])):
        avgList[i][j] /= 19
##************************************************************************************************



##************************************************************************************************
##Ranking documents
##************************************************************************************************
def ranking(avgList):
    hm = {}
    ht = []
    hm1 = {}
    index = 1
    for i in avgList:
        hm[harmonic_mean(i)]=index
        ht.append(harmonic_mean(i))
        index += 1
    for key in sorted(hm):
        hm1[key] = hm[key]
    return hm1,ht
##************************************************************************************************



##************************************************************************************************
## Results
##************************************************************************************************
#Dict of ranked documents with harmonic mean
result,grapharr = ranking(avgList)

#List of ranked documents
ranked_list = []
ranked_list = result.values()

#Printing ranked documents list
print(ranked_list)
##************************************************************************************************



##************************************************************************************************
##Graphs
##************************************************************************************************
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline

#Intensity of heatmap
intensity = avgList
intensity = np.array(intensity)

#Plotting heatmap between documents based on document similarity
fig = plt.figure()
fig, ax = plt.subplots(1,1, figsize=(12,12))
heatplot = ax.imshow(intensity, cmap='BuPu')
ax.set_title("Similarity between documents")
ax.set_xlabel('Document1')
ax.set_ylabel('Document2')
plt.show()
#Normalizing result
res = [i/sum(grapharr) for i in grapharr]

#Plotting histogram on results
a = np.array(res)
plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of cohesion")
plt.show()


#Plotting Heatmap of Structured Based Similarity
sbs = simMat[18]
sbs = np.array(sbs)
fig = plt.figure()
fig, ax = plt.subplots(1,1, figsize=(12,12))
heatplot = ax.imshow(sbs, cmap='BuPu')
ax.set_title("Stuctural similarity between documents")
ax.set_xlabel('Document1')
ax.set_ylabel('Document2')
plt.show()
##************************************************************************************************