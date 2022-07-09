import numpy as np
import random
from pyrsgis import raster
from pyrsgis.ml import array_to_chips   #imagechipsfromarray doesnt work anymore.
from math import floor

class multiclass_preprocess:

    def __init__(self, feature_file, positivelabel_file):
        self.feature_file =  feature_file
        self.positivelabel_file = positivelabel_file

    def read_and_normalise_input(self):
        # Reading and normalizing input data
        dsFeatures, arrFeatures = raster.read(self.feature_file, bands='all')
        # arrFeatures = arrFeatures.astype(float)
        arrFeatures = np.float16(arrFeatures)

        for i in range(arrFeatures.shape[0]):
            bandMin = arrFeatures[i][:][:].min()
            bandMax = arrFeatures[i][:][:].max()
            bandRange = bandMax-bandMin
            for j in range(arrFeatures.shape[1]):
                for k in range(arrFeatures.shape[2]):
                    arrFeatures[i][j][k] = (arrFeatures[i][j][k]-bandMin)/bandRange

        # Creating chips using pyrsgis
        features = array_to_chips(arrFeatures, x_size=7, y_size=7)

        return features

    def read_and_normalise_labels(self):
        # Reading and reshaping the label file
        dsPositive, positiveLabels = raster.read(self.positivelabel_file)

        # Generating random samples
        nonZero_count = 0

        for i in range(positiveLabels.shape[0]):
            for j in range(positiveLabels.shape[1]):
                if positiveLabels[i,j] != 0:
                    nonZero_count += 1

        k = 0
        index = np.zeros(((positiveLabels.shape[0]*positiveLabels.shape[1])-nonZero_count,2))   #Gives the number of zero labels

        #Appending all the coordinates of points with zero labels in the index array
        for i in range(positiveLabels.shape[0]):
            for j in range(positiveLabels.shape[1]):
                if positiveLabels[i,j] == 0:
                    index[k,0] = i
                    index[k,1] = j
                    k += 1

        #Random Index of nonZero_count/2 coordinates from index
        randomIndex = random.sample(range(index.shape[0]), floor(nonZero_count/2))
        negativeLabels = np.zeros(positiveLabels.shape)

        index = index.astype(int)

        for i in range(len(randomIndex)):
            negativeLabels[index[randomIndex[i],0],index[randomIndex[i],1]] = np.max(positiveLabels)+1

        del index


        positiveLabels = positiveLabels.flatten()
        negativeLabels = negativeLabels.flatten()

        features = self.read_and_normalise_input()
        # Separating and balancing the classes
        positiveFeatures = features[positiveLabels!=0]
        positiveLabels = positiveLabels[positiveLabels!=0]

        negativeFeatures = features[negativeLabels==np.max(positiveLabels)+1]
        negativeLabels = negativeLabels[negativeLabels==np.max(positiveLabels)+1]

        # Combining the balanced features
        features = np.concatenate((positiveFeatures, negativeFeatures), axis=0)
        labels = np.concatenate((positiveLabels, negativeLabels), axis=0)

        del positiveFeatures
        del negativeFeatures
        del positiveLabels
        del negativeLabels

        return features, labels

    # Defining the function to split features and labels
    def train_test_split(self, trainProp=0.75):
        features, labels = self.read_and_normalise_labels()
        dataSize = features.shape[0]
        sliceIndex = int(dataSize*trainProp)
        randIndex = np.arange(dataSize)
        random.shuffle(randIndex)
        train_x = features[[randIndex[:sliceIndex]], :, :, :][0]
        test_x = features[[randIndex[sliceIndex:]], :, :, :][0]
        train_y = labels[randIndex[:sliceIndex]]
        test_y = labels[randIndex[sliceIndex:]]

        train_y[train_y == 3] = 0
        test_y[test_y == 3] = 0
        return(train_x, train_y, test_x, test_y)