import numpy as np
import math
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker():
    def __init__(self, maxDisappeared=20):

        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
    def register(self, centroid):

        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):

        del self.objects[objectID]
        del self.disappeared[objectID]
        self.dID.append(objectID)

    def update(self, rects):
        self.dID = []
        if len(rects) == 0:

            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects,self.dID


        inputCentroids = np.zeros((len(rects), 6), dtype="int")


        for (i, (startX, startY, endX, endY)) in enumerate(rects):

            cX=int((startX + endX) / 2.0)
            cY=int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY,startX,startY,endX,endY)


        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:

            objectIDs = list(self.objects.keys())
            t = list(self.objects.values())
            objectCentroids=[]
            inc=[]
            for i in t:
                objectCentroids.append(i[:2])
            for i in inputCentroids:
                inc.append(i[:2])

#            print(np.array(objectCentroids))
#            print(np.array(inc))
            D = dist.cdist(np.array(objectCentroids), np.array(inc))
#            print(D)
            rows = D.min(axis=1).argsort()
#            print(rows)
            cols= D.argmin(axis=1)[rows]
#            print(cols)
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                objectID= objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)
#            print(usedRows)
#            print(usedCols)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
#            print(unusedRows)
#            print(unusedCols)
            if D.shape[0] >= D.shape[1]:

                for row in unusedRows:

                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects,self.dID
