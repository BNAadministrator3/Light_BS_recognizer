# -*- coding: utf-8 -*-

"""
Description: To access the labels in .json file.
License_info: GPL
what to learn: os.remove; dump vs dumps are like the matlab's jsondecode
"""
import os
import json
from random import shuffle,randint
from keras.utils import to_categorical
from help_func.feature_transformation import *
from help_func.utilities import CLASS_NUM

def querylabel(ranges, form):
    assert (isinstance(ranges, tuple))
    assert (len(ranges) == 2)
    begining, ending = ranges
    assert (isinstance(begining, int))
    assert (isinstance(ending, int))
    # form = self.labeltable[filename]
    value = None
    for items in form:
        if begining >= items[0] and ending <= items[1]:
            value = items[2]
            break
    if value != None:
        return value
    else:
        print('[*ERROR] Sth wrong!')
        assert (0)
class dataset_operator():
    def __init__(self,labelpath=None,foldpath=None):
        self.labeltable = self.readlabels(labelpath)
        self.datatable = self.loaddatadirectory(foldpath)

    def readlabels(self,jpath=None):
        if jpath is None:
            jpath = os.path.join(os.getcwd(), 'indices', 'content.json')
        else:
            assert(os.path.exists(jpath))
            assert(os.path.isfile(jpath))
        with open(jpath) as json_labels:
            d = json.load(json_labels)
        lookuptable = {}
        for items in d:
            name = items[0]
            labels = []
            for units in items[1]['_TableRecords_']:
                unit = (units[0][0], units[0][1], units[1][0])
                labels.append(unit)
            lookuptable[name] = labels
        return lookuptable

    def loaddatadirectory(self,dpath=None):
        if dpath is None:
            dpath = os.path.join( os.getcwd(), 'data')
        assert (os.path.exists(dpath))
        assert (os.path.isdir(dpath))
        dname = os.listdir(dpath)
        dcpath = [os.path.join(dpath,i) for i in dname]
        datatable = tuple(zip(dname,dcpath))
        return datatable
        #thw output is a nested tuple.

#target: as soon as possible and use the generator+one-step+tb method to monitor other metrics
class buildTrainValtest():
    def __init__(self,datatable,labeltable):
        self.datatable = datatable
        self.labeltable = labeltable
        print()
    # all about the baseline !!!!!!!!!!!!!
    # start from the end of the last work
    # isolated approaches
    # generator+one-step+tb method
    # oversample for imbalanced
    #
    #
    def cutintopiecesLoo(self,flag,reserve=False):
        '''
        This function is just form the base of the training,validation and testing data
        :param flag:
        :param ddir:
        :return: training data plus labels, validation data plus labels and testing data plus labels
        '''
        shift = 0.1 # At present, shift should keep equal to wlen otherwise the last pieces is shorter than else.
        assert(flag in range(len(self.datatable)))
        #1. test set
        TestFnames = [self.datatable[flag][0]]
        testFpaths = [self.datatable[flag][1]]
        testRecording = []
        for path,name in zip(testFpaths,TestFnames):
            wavsignal, fs = read_wav_data(path)
            assert(wavsignal.shape[1]==fs*60)
            image = SimpleMfccFeatures(wavsignal, fs, shift)
            pwidth = int(1 / shift)
            dataum = []
            tagum = []
            for i in range(60):
                piece = image[i*pwidth:(i+1)*pwidth,:]
                if reserve is False:
                    piece = np.squeeze(piece.reshape(1,-1))
                else:
                    piece = piece.reshape(1,piece.shape[0],piece.shape[1])
                dataum.append(piece)
                tag = querylabel((i,i+1),self.labeltable[name])
                assert(tag in ('Present','Absent','Weak','Uncertain'))
                inmaplist={'Present':0,'Absent':1,'Weak':1,'Uncertain':1}
                tagum.append(inmaplist[tag])
            testRecording.append((dataum,tagum))
        self.looTestMark = TestFnames[0]
        del TestFnames, testFpaths
        #2. training set
        trainRecordings = []
        ls = list(range(len(self.datatable)))
        ls.remove(flag)
        for i in ls:
            trainFnames = [self.datatable[i][0]]
            trainFpaths = [self.datatable[i][1]]
            dataum = []
            tagum = []
            for path, name in zip(trainFpaths, trainFnames):
                wavsignal, fs = read_wav_data(path)
                assert (wavsignal.shape[1] == fs * 60)
                image = SimpleMfccFeatures(wavsignal, fs, shift)
                pwidth = int(1 / shift)
                for i in range(60):
                    piece = image[i * pwidth : (i + 1) * pwidth, :]
                    if reserve is False:
                        piece = np.squeeze(piece.reshape(1, -1))
                    else:
                        piece = piece.reshape(1, piece.shape[0], piece.shape[1])
                    dataum.append(piece)
                    tag = querylabel((i, i + 1), self.labeltable[name])
                    assert (tag in ('Present', 'Absent', 'Weak', 'Uncertain'))
                    inmaplist = {'Present': 0, 'Absent': 1, 'Weak': 1, 'Uncertain': 1}
                    tagum.append(inmaplist[tag])
                trainRecordings.append((dataum,tagum))
        shuffle(trainRecordings)
        #join together
        testoverall = testRecording
        trainoverall = trainRecordings
        self.trainRecordings = trainRecordings
        self.trainList = trainoverall
        self.testList = testoverall

    def getNonRepetitiveData(self,n_start, type='train'):
        assert(isinstance(n_start,int))
        assert(type in ('train','test'))
        dataChecked = self.testList if type == 'test' else self.trainList
        num = len(dataChecked)
        data_input, data_label = dataChecked[n_start%num]
        data_input = np.array(data_input)
        data_label = np.array(data_label)
        return data_input, data_label

    def generatorSetting(self,batch_size=32):
        #Based on the train_on_batch, the batch size is not fixed! Yeah it is.
        assert(batch_size <= len(self.trainRecordings))
        #1. get the counts of training data
        if len(self.trainRecordings)%batch_size ==0:
            iteration_per_epoch =  len(self.trainRecordings)//batch_size
        else:
            iteration_per_epoch = len(self.trainRecordings) // batch_size + 1
        self.iteration_per_epoch = iteration_per_epoch
        self.batch_szie = batch_size

    def __getData__(self, n_start):  # Due to the class weight, samples in every batch does not need to be class-equal.
        assert(isinstance(n_start,int))
        assert(n_start<=self.iteration_per_epoch)
        if n_start<self.iteration_per_epoch:
            bdata = self.trainRecordings[(n_start-1)*self.batch_szie: n_start*self.batch_szie]
        else:
            bdata = self.trainRecordings[(n_start - 1) * self.batch_szie:]
        shuffle(bdata)
        data_input,data_label = zip(*bdata)
        return data_input, data_label

    def data_genetator(self): #This generator is only in charge of one single epoch
        shuffle(self.trainRecordings)
        while True:
            ran_num = randint(1, self.iteration_per_epoch)  # 获取一个随机数
            X,y = self.__getData__(n_start=ran_num)

            X = np.array(X)
            y = np.array(y)
            yield X, to_categorical(y, num_classes=CLASS_NUM)  # 功能只是转成独热编码
        pass

    def __setCounting__(self,dataset):
        assert(isinstance(dataset,dict))
        assert(all([i in ('Present', 'Absent', 'Weak', 'Uncertain') for i in dataset.keys()]))
        return [len(dataset['Present']), len(dataset['Absent']), len(dataset['Weak']),len(dataset['Uncertain'])]


if __name__ == '__main__':
    ##test the first class
    # jpath = os.path.join(os.getcwd(), 'indices', 'content.json')
    # dpath = os.path.join(os.getcwd(), 'data')
    # opit = dataset_operator(jpath)
    # # print(opit.querylabel('huangjun-bowel08211.wav',(59,60)))
    # opit.loaddatadirectory(dpath)
    # opit.write2json()

    #test feature abstraction
    # path = os.path.join( os.getcwd(), 'data','f_dingcong-bowel08201.wav' )
    # wavsignal, fs = read_wav_data(path)

    #Test jointly

    opit = dataset_operator()
    shifter = buildTrainValtest(opit.datatable,opit.labeltable)
    shifter.cutintopiecesLoo(flag=3)
    # shifter.functionCheck()
    # print(shifter.setCounting(shifter.trainPieces))
    # print(shifter.getData())
    shifter.generatorSetting(batch_size=16)
    yielddata = shifter.data_genetator()
    for i in yielddata:
        qq,dd = i
        a=1