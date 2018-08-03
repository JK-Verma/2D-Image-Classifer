############################################################################################
# Title:         AA3PG 2D STL Classifier Trainer
# Description:   Trains the AA3PG 2D Images Classifier on a data set of your choice.
# Configuration: data/confs.json
# Last Modified: 2018-08-04
############################################################################################

import os, glob, random, cv2, caffe, lmdb, json
import numpy as np
from caffe.proto import caffe_pb2

class Train():
    
    def __init__(self):
                
        self._confs = {}
        self.photoPaths = []
        self.directories = []
        self.trainSize = 0
        self.validationSize = 0
        self.testSize = 0
        self.in_db = None
        
        with open('data/confs.json') as confs:

            self._confs = json.loads(confs.read())
        
        os.system('rm -rf  ' + self._confs["ClassifierSettings"]["train_lmdb"])
        os.system('rm -rf  ' + self._confs["ClassifierSettings"]["validation_lmdb"])

    def sortData(self):

        for dirName in os.listdir(self._confs["ClassifierSettings"]["dataset_dir"]):

            path = os.path.join(self._confs["ClassifierSettings"]["dataset_dir"], dirName)

            if os.path.isdir(path):

                self.directories.append(path)
                
        print(self.directories)

        for directory in self.directories:

            for filename in os.listdir(directory):

                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.gif'):

                    path = os.path.join(directory, filename)
                    self.photoPaths.append(path)

                    print(filename)
                    print("-- Image Added For Training")
                    print("")

                else:

                    print(filename)
                    print("-- Invalid Image Skipped")
                    print("")
                    continue

    def createLMDB(self):

        random.shuffle(self.photoPaths)
        self.in_db = lmdb.open(self._confs["ClassifierSettings"]["train_lmdb"], map_size=int(1e12))

        self.trainSize = ((len(self.photoPaths) * self._confs["ClassifierSettings"]["trainCut"]) / 100)
        self.validationSize = ((len(self.photoPaths) * self._confs["ClassifierSettings"]["trainCut"]) / 100)
        self.testSize = ((len(self.photoPaths) * self._confs["ClassifierSettings"]["testCut"]) / 100)

        print("Data Size: "+str(len(self.photoPaths)))
        print("Photos Size: "+str(self.trainSize))
        print("Validation Size: "+str(self.validationSize))
        print("Test Size: "+str(self.testSize))
        
        # To be updated
        
Train = Train()
Train.sortData()
Train.createLMDB()
print("")