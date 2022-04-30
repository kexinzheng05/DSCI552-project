import numpy as np
import math
import pandas as pd
import sys
from skimage.transform import resize
from skimage.feature import hog
import cv2
from sklearn.cluster import KMeans


class FastMap():
    def __init__(self, symDistance):
        self.symDistance = symDistance
        self.farthest = []

    # Given the distance of objects and the number of them, find the farthest pair of objects
    def farthestImages(self, numImages):
        start = np.random.randint(numImages)
        farthest = np.argmax(self.symDistance[start])
        newFarthest = np.argmax(self.symDistance[farthest])
        while newFarthest != start:
            start, farthest = farthest, newFarthest
            newFarthest = np.argmax(self.symDistance[farthest])
        return start, farthest

    # Calculate one coordinate of object i according to the information of the farthest objects a & b
    def calCoordinate(self, imageA, imageB, imageI):
        dist_AI = self.symDistance[imageA][imageI]
        dist_BI = self.symDistance[imageB][imageI]
        dist_AB = self.symDistance[imageA][imageB]
        # print(dist_AI, dist_BI, dist_AB)
        coordinate = (dist_AI ** 2 + dist_AB ** 2 - dist_BI ** 2) / (2 * dist_AB)  # first coordinate of object i
        return coordinate

    # Recalculate the distance of objects after embedding, return a new distance matrix
    def reCalDistance(self, Coordinates, imageA, imageB, preA, preB):
        numImage = len(Coordinates)
        distanceMatrix = np.zeros(self.symDistance.shape)
        for i in range(numImage):
            for j in range(numImage):
                if i != j:
                    #print(i, j, imageA, imageB, preA, preB)
                    #print(symDistance[i][j], Coordinates[i], Coordinates[j])
                    # if symDistance[i][j] == 0:
                    if (i == preA and j == preB) or (i == preB and j == preA) or (i == imageA and j == imageB) or (i == imageB and j == imageA):
                        distance = 0.0
                    else:
                        distance = math.sqrt(self.symDistance[i][j] ** 2 - (Coordinates[i] - Coordinates[j]) ** 2)
                    distanceMatrix[i][j] = distance
                    distanceMatrix[j][i] = distance
        self.symDistance = distanceMatrix

    # Given the farthest images A & B, return the coordinate of all the objects
    def getCoordinates(self, imageA, imageB):
        Coordinates = []
        for image in range(len(self.symDistance)):
            # print(imageA, imageB, image)
            coord = self.calCoordinate(imageA, imageB, image)
            Coordinates.append(coord)
        return Coordinates

    # A FastMap algorithm
    # Obtain the new distance matrix and return the coordinates in 2D space
    def fit(self, numImages, numDim):
        coord2D = [[] for i in range(numImages)]
        preFarthest = [[-1, -1]]
        for i in range(numDim):
            #print("\niteration", i)
            imageA, imageB = self.farthestImages(numImages)
            #print(imageA, imageB)
            preFarthest.append([imageA, imageB])
            coordinates = self.getCoordinates(imageA, imageB)
            [coord2D[i].append(coordinates[i]) for i in range(len(coordinates))]

            self.reCalDistance(coordinates, imageA, imageB, preFarthest[i][0], preFarthest[i][1])
        return coord2D

# calculate distance of hog image
def calDistance(image1, image):
    distance = np.sqrt(np.sum((image1-image)**2, axis=1))
    return distance

def main():
    np.set_printoptions(threshold=sys.maxsize)

    dtype_dic = {'article_id': str}
    articles = pd.read_csv('D:/Download/h-and-m-personalized-fashion-recommendations/articles.csv', dtype=dtype_dic, usecols=[0, 4])
    articles = articles.set_index("article_id")["product_type_name"].to_dict()
    #print(articles)
    article_type = {}
    for articleId, articleType in articles.items():
        article_type.setdefault(articleType, []).append(articleId)
    print("len(article_type)", len(article_type))

    types = list(article_type.keys())
    #for oneType in article_type:
    for i in range(len(article_type)):
        oneType = types[i]
        print(oneType)
        articleIds = article_type[oneType]
        print(len(articleIds))
        if len(articleIds) < 10:
            continue
        images = []

        for articleId in articleIds:
            path = "D:/Download/h-and-m-personalized-fashion-recommendations/images/%s/%s.jpg" % (
            articleId[:3], articleId)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = image.astype(np.float32)
                # image = Image.open(path)
                resized_img = resize(image, (128, 64))
                hogimage = hog(resized_img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
                images.append(hogimage)

        images = np.array(images)
        #print(images.shape)

        numImage = len(images)
        symDistance = np.zeros((numImage, numImage))
        # using edge descriptors firstly
        for i in range(numImage):
            if i == numImage - 1:
                continue
            distance = calDistance(images[i], images[i + 1:])
            symDistance[i, i + 1:] = distance
            symDistance[i + 1:, i] = distance
        fastMap = FastMap(symDistance)
        coord2D = fastMap.fit(numImage, numDim=2)
        coord2D = np.array(coord2D)

        if numImage > 1000:
            numCluster = numImage//300+1
        else:
            numCluster = 3
        kmean = KMeans(n_clusters=numCluster, max_iter=1000).fit(coord2D)
        labels = kmean.labels_
        df = pd.DataFrame({"article_id": articleIds, "cluster": labels})
        oneType = oneType.replace("/", " ")
        df.to_csv(f"{oneType}.csv")

if __name__ == "__main__":
    main()