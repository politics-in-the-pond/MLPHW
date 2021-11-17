import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from pyclustering.cluster.clarans import clarans
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def getScore(data, result):
    try:
        score = silhouette_score(data, result)
    except ValueError as e:
        score = -1
        pass
    return score

def showHitmap(data):
    corr = data.corr();corr

    plt.figure(figsize = (20,15))

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(data = corr,
            annot = True,
            mask = mask,
            fmt = '.2f',
            linewidths = 1.,
            cmap = 'RdYlBu_r')
    plt.show()

def showScatter(data, label, x_loc, y_loc):
    plt.scatter(data.iloc[:,x_loc], data.iloc[:,y_loc], c=label)
    plt.show()

def findBestParams(data, Encoders, Scalars, ClusterParams):

    threads = 16

    scores = []
    labels = []
    bestScore = -100.0
    bestResult = 0
    bestIndex = 0

    encodeTarget = data['ocean_proximity']
    encodeTargetarr = encodeTarget.to_numpy()
    data = data.drop(['ocean_proximity', 'median_house_value'], axis = 1)

    for e in Encoders:
        encoderLabel = "encoder : " + str(e)
        if e != 'None':
            encoder = e
            encoder.fit(encodeTargetarr)
            encodeTargetarr = encoder.transform(encodeTargetarr)
            encoded = pd.DataFrame(encodeTargetarr)
            encoded = pd.concat([data, encoded], axis=1)
            encodeTargetarr = encodeTargetarr.reshape(-1, 1)
            encoded = encoded.dropna(axis=0)
        else:
            encoded = data
            encoded = encoded.dropna(axis=0)

        for s in Scalars:
            scalar = s
            encoded = scalar.fit_transform(encoded)

            scalarLabel = ", scaler : " + str(s)

            for n_clusters in ClusterParams[0]['n_clusters']:
                for algorithm in ClusterParams[0]['algorithm']:
                    for init in ClusterParams[0]['init']:
                        paramLabel = ", method : KMeans, n_cluster : " + str(n_clusters) + ", algorithm : " + str(algorithm) + ", init : " + str(init)
                        
                        km = KMeans(n_clusters=n_clusters, algorithm=algorithm, init=init)
                        result = km.fit_predict(encoded)
                        score = getScore(encoded, result)

                        label = encoderLabel + scalarLabel + paramLabel + ", score :" + str(score)
                        print(label)

                        scores.append(score)
                        labels.append(label)

                        if score > bestScore:
                            bestScore = score
                            bestResult = result
                            bestIndex = len(scores) - 1

            for eps in ClusterParams[1]['eps']:
                for algorithm in ClusterParams[1]['algorithm']:
                    for leaf_size in ClusterParams[1]['leaf_size']:
                        for min_samples in ClusterParams[1]['min_samples']:

                            paramLabel = ", method : DBSCAN , eps : " + str(eps) + ", algorithm : " + str(algorithm) + ", leaf_Size : " + str(leaf_size) + ", min_samples : " + str(min_samples)

                            db = DBSCAN(eps=eps, algorithm=algorithm, leaf_size=leaf_size, min_samples=min_samples)
                            result = db.fit_predict(encoded)
                            score = getScore(encoded, result)

                            label = encoderLabel + scalarLabel + paramLabel + ", score :" + str(score)
                            print(label)

                            scores.append(score)
                            labels.append(label)

                            if score > bestScore:
                                bestScore = score
                                bestResult = result
                                bestIndex = len(scores) - 1
            

            for n_components in ClusterParams[2]['n_components']:
                for convariance_type in ClusterParams[2]['convariance_type']:
                    for init_params in ClusterParams[2]['init_params']:

                        paramLabel = ", method : GaussianMixture, n_components : " + str(n_components) + ", convariance_type : " + str(convariance_type) + ", init_params : " + str(init_params)

                        gm = GaussianMixture(n_components=n_components, covariance_type=convariance_type, init_params=init_params)
                        result = gm.fit_predict(encoded)
                        score = getScore(encoded, result)
                        
                        label = encoderLabel + scalarLabel + paramLabel + ", score :" + str(score)
                        print(label)

                        scores.append(score)
                        labels.append(label)

                        if score > bestScore:
                            bestScore = score
                            bestResult = result
                            bestIndex = len(scores) - 1


            
            for number_clusters in ClusterParams[3]['number_clusters']:
                for numlocal in ClusterParams[3]['numlocal']:
                    for maxneighbor in ClusterParams[3]['maxneighbor']:

                        paramLabel = ", method : CLARANS, number_clusters : " + str(number_clusters) + ", numlocal : " + str(numlocal) + ", maxneighbor : " + str(maxneighbor)

                        cl = clarans(encoded, number_clusters, numlocal, maxneighbor)
                        result = cl.process()
                        score = silhouette_score(encoded, result.get_cluster_encoding)

                        label = encoderLabel + scalarLabel + paramLabel + ", score :" + str(score)
                        print(label)

                        scores.append(score)
                        labels.append(label)

                        if score > bestScore:
                            bestScore = score
                            bestResult = result
                            bestIndex = len(scores) - 1
            
            
            for bandwidth in ClusterParams[4]['bandwidth']:
                
                paramLabel = ", method : MeanShift, bandwidth : " + str(bandwidth)

                ms = MeanShift(bandwidth=bandwidth, n_jobs=threads)
                result = ms.fit_predict(encoded)
                score = getScore(encoded, result)

                label = encoderLabel + scalarLabel + paramLabel + ", score :" + str(score)
                print(label)

                scores.append(score)
                labels.append(label)

                if score > bestScore:
                    bestScore = score
                    bestResult = result
                    bestIndex = len(scores) - 1
            
    return scores, encoded, labels, bestScore, bestResult, bestIndex

def main():
    np.set_printoptions(threshold=sys.maxsize)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    tmpdf = pd.read_csv(".\housing.csv")

    
    Scalars = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    ClusterParams = []
    kmeansParams = {
        'n_clusters' : [2, 4, 6, 8, 10, 12],
        'algorithm' : ['auto', 'full', 'elkan'],
        'init' : ['k-means++', 'random']
    }
    
    dbscanParams = {
        'eps' : [0.5, 1.0],
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size' : [3, 5, 15, 20],
        'min_samples' : [3, 5, 10, 15]
    }

    GMParams = {
        'n_components' : [2, 4, 6, 8, 10, 12],
        'convariance_type' : ['full', 'tied', 'diag', 'spherical'],
        'init_params' : ['kmeans', 'random']
    }

    claransParams = {
        'number_clusters' : [2, 4, 6, 8, 10, 12],
        'numlocal' : [2, 4, 8, 10],
        'maxneighbor' : [3, 5, 15]
    }

    MSParams = {
        'bandwidth' : [0.7, 1.3, 2.576979121414909, 5]
    }

    ClusterParams.append(kmeansParams)
    ClusterParams.append(dbscanParams)
    ClusterParams.append(GMParams)
    ClusterParams.append(claransParams)
    ClusterParams.append(MSParams)

    #do with high correlation features

    new_tmpdf = tmpdf.drop(['households', 'longitude', 'latitude', 'median_income'], axis = 1)

    Encoders = [LabelEncoder(), OneHotEncoder(sparse=False)]

    scores, encoded, labels, bestScore, bestResult, bestIndex = findBestParams(new_tmpdf, Encoders, Scalars, ClusterParams)

    print("best params = " + str(labels[bestIndex]))

    showScatter(tmpdf, bestResult)


    #do with high correlation features

    new_tmpdf = tmpdf.drop(['housing_median_age', 'longitude', 'latitude', 'median_income'], axis = 1)

    Encoders = ['None']

    scores, encoded, labels, bestScore, bestResult, bestIndex = findBestParams(new_tmpdf, Encoders, Scalars, ClusterParams)

    print("best params = " + str(labels[bestIndex]))

    showScatter(tmpdf, bestResult)


    #do with high correlation features + direction of houses

    new_tmpdf = tmpdf.drop(['housing_median_age', 'median_income'], axis = 1)

    Encoders = [LabelEncoder(), OneHotEncoder(sparse=False)]

    scores, encoded, labels, bestScore, bestResult, bestIndex = findBestParams(new_tmpdf, Encoders, Scalars, ClusterParams)

    print("best params = " + str(labels[bestIndex]))


    #do with all features

    Encoders = [LabelEncoder(), OneHotEncoder(sparse=False)]

    scores, encoded, labels, bestScore, bestResult, bestIndex = findBestParams(tmpdf, Encoders, Scalars, ClusterParams)

    print("best params = " + str(labels[bestIndex]))

main()
