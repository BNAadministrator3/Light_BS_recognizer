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

CLASS_NUM = 2

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
            # jpath = os.path.join(os.getcwd(), 'indices', 'content-2nditeration.json')
            jpath = os.path.join(os.getcwd(), 'indices', 'content.json')
            print()
            # print('Path Mark: content-2nditeration.json')
            print('Path Mark: content.json')
            print()
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
        #Will remove the unnecessary items
        # target = 'f_fengfriend1-bowel08201.wav'
        # self.labeltable.pop(target)
        # # # print(pop_obj)
        # temp = list(self.datatable)
        # for tp in temp:
        #     if tp[0] == target:
        #         temp.remove(tp)
        #         break
        # del self.datatable
        # self.datatable = tuple(temp)

    # all about the baseline !!!!!!!!!!!!!
    # start from the end of the last work
    # isolated approaches
    # generator+one-step+tb method
    # oversample for imbalanced
    #
    #
    def cutintopieces(self,flag):
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
        testPieces = {}
        testPieces['Present'] = []
        testPieces['Absent'] = []
        testPieces['Weak'] = []
        testPieces['Uncertain'] = []
        for path,name in zip(testFpaths,TestFnames):
            wavsignal, fs = read_wav_data(path)
            assert(wavsignal.shape[1]==fs*60)
            image = SimpleMfccFeatures(wavsignal, fs, shift)
            pwidth = int(1 / shift)
            for i in range(60):
                piece = image[i*pwidth:(i+1)*pwidth,:]
                piece = piece.reshape(piece.shape[0], piece.shape[1], 1)
                tag = querylabel((i,i+1),self.labeltable[name])
                assert(tag in ('Present','Absent','Weak','Uncertain'))
                testPieces[tag].append((piece,tag,(i,i+1)))
        self.looTestMark = TestFnames[0]
        del TestFnames, testFpaths
        #2. training set
        trainPieces = {}
        trainPieces['Present'] = []
        trainPieces['Absent'] = []
        trainPieces['Weak'] = []
        trainPieces['Uncertain'] = []
        ls = list(range(len(self.datatable)))
        ls.remove(flag)
        for i in ls:
            trainFnames = [self.datatable[i][0]]
            trainFpaths = [self.datatable[i][1]]
            for path, name in zip(trainFpaths, trainFnames):
                wavsignal, fs = read_wav_data(path)
                assert (wavsignal.shape[1] == fs * 60)
                image = SimpleMfccFeatures(wavsignal, fs, shift)
                pwidth = int(1 / shift)
                for i in range(60):
                    piece = image[i * pwidth : (i + 1) * pwidth, :]
                    piece = piece.reshape(piece.shape[0],piece.shape[1],1)
                    self.pieceShape = piece.shape
                    tag = querylabel((i, i + 1), self.labeltable[name])
                    assert (tag in ('Present', 'Absent', 'Weak', 'Uncertain'))
                    trainPieces[tag].append((piece, tag, (i,i+1)))
        for i in ('Present', 'Absent', 'Weak', 'Uncertain'):
            shuffle(trainPieces[i])
            shuffle(testPieces[i])
        testcnts = self.setCounting(testPieces)
        #join together
        testoverall = []
        trainoverall = []
        for i in ('Present', 'Absent', 'Weak', 'Uncertain'):
            testoverall = testoverall + testPieces[i]
            trainoverall = trainoverall + trainPieces[i]
        assert(len(testoverall) == sum(testcnts))
        shuffle(testoverall)
        self.trainPieces = trainPieces
        self.trainList = trainoverall
        self.testList = testoverall

    def getNonRepetitiveData(self,n_start, type='train', mark=False):
        assert(isinstance(n_start,int))
        assert(type in ('train','test'))
        dataChecked = self.testList if type == 'test' else self.trainList
        num = len(dataChecked)
        data_input, data_label, stamp = dataChecked[n_start%num]
        maplist = {'Present': 0, 'Absent': 1, 'Weak': 1, 'Uncertain': 1}
        temp = data_label
        data_label = maplist[data_label]
        if mark is False:
            return data_input, data_label
        else:
            return data_input, data_label, stamp, temp

    def generatorSetting(self,batch_size=32):
        #1. get the counts of training data
        traincnts = self.setCounting(self.trainPieces)
        assigncnts = [ myround(i/float(sum(traincnts))*batch_size) for i in traincnts]
        if sum(assigncnts) == batch_size:
            pass
        else:
            assert(abs(sum(assigncnts)-batch_size)==1)
            if sum(assigncnts)>batch_size:
                assigncnts[0] = assigncnts[0] - 1
            else:
                assigncnts[0] = assigncnts[0] + 1
        iteration_per_epoch =  min([aggrega//single for aggrega,single in zip(traincnts, assigncnts)])
        # traincnts[0] = traincnts[0] + traincnts[2]
        # traincnts[1] = traincnts[1] + traincnts[3]
        traincnts[1] = traincnts[1] + traincnts[2] + traincnts[3]
        del traincnts[2:4]
        classWeights = [sum(traincnts) / (CLASS_NUM * el) for el in traincnts]
        self.sampleoffset = assigncnts
        self.iteration_per_epoch = iteration_per_epoch
        self.classWeights = classWeights

    def getData(self, n_start):  # Due to the class weight, samples in every batch does not need to be class-equal.
        assert(isinstance(n_start,int))
        data_input = []
        data_label = []
        for tag,offset in zip(('Present', 'Absent', 'Weak', 'Uncertain'),self.sampleoffset):
            begining = n_start * offset
            temp = self.trainPieces[tag][begining:begining+offset]
            tempunzip = list(map(list,zip(*temp)))
            data_input = data_input + tempunzip[0]
            data_label = data_label + tempunzip[1]
        together = list(zip(data_input,data_label))
        shuffle(together)
        data_input,data_label = zip(*together)
        maplist = { 'Present':0, 'Absent':1, 'Weak':1, 'Uncertain':1 }
        data_label = [maplist[i] for i in data_label]
        return data_input, data_label

    def data_genetator(self): #This generator is only in charge of one single epoch
        for i in ('Present', 'Absent', 'Weak', 'Uncertain'):
            shuffle(self.trainPieces[i])
        while True:
            ran_num = randint(0, self.iteration_per_epoch-1)  # 获取一个随机数
            X,y = self.getData(n_start=ran_num)

            # from collections import Counter
            # a = list(Counter(y).items())

            X = np.array(X)
            y = np.array(y)
            yield X, to_categorical(y, num_classes=CLASS_NUM)  # 功能只是转成独热编码
        pass

    def setCounting(self,dataset):
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
    shifter.cutintopieces(flag=3)
    # shifter.functionCheck()
    # print(shifter.setCounting(shifter.trainPieces))
    # print(shifter.getData())
    shifter.generatorSetting(batch_size=16)
    yielddata = shifter.data_genetator()
    for i in yielddata:
        qq,dd = i
        a=1