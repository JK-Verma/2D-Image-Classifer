############################################################################################
# Title:         AA3PG Caffe Helper
# Description:   Helper functions for Caffe.
# Configuration: data/confs.json
# Last Modified: 2018-09-04
############################################################################################

import cv2, json
from caffe.proto import caffe_pb2

class CaffeHelper():
    
    def __init__(self):

        self._confs = {}
        
        with open('data/confs.json') as confs:

            self._confs = json.loads(confs.read())
            
    def transform(self, img):
        
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        
        return cv2.resize(img,(self._confs["ClassifierSettings"]["imageWidth"],self._confs["ClassifierSettings"]["imageHeight"]),interpolation = cv2.INTER_CUBIC)

    def createDatum(self, imageData, label):
    
        datum = caffe_pb2.Datum()
        datum.channels = imageData.shape[2]
        datum.height = imageData.shape[0]
        datum.width = imageData.shape[1]
        datum.data = imageData.tobytes()
        datum.label = int(label)

        return datum