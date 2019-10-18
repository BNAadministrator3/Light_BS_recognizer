from keras import optimizers
import tensorflow as tf
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.layers import *
import keras.backend as k
import os
from tqdm import tqdm
from random import randint
import time
import importlib
import gc

# from A1c_leaveOneOutData import dataset_operator,  buildTrainValtest, CLASS_NUM
from A1b_binaryDataSpeech import dataset_operator,  buildTrainValtest
from help_func.utilities import *
from help_func.evaluation import Compare4, plot_confusion_matrix
from A3a_printLabels import misclassifiedManagement

def clrdir(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            clrdir(c_path)
        else:
            os.remove(c_path)

class Networks():
    def __init__(self,inputShape):
        print("Let's begin networking!")
        # input_shape = (AUDIO_LENGTH, 26, 1) (10,26,1)
        self.model_input = Input(shape=inputShape)

    def CreateRegularCNNModel(self):
        level_h1 = ReguBlock(32)(self.model_input)  # 卷积层
        level_m1 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h1)  # 池化层
        level_h2 = ReguBlock(64)(level_m1)
        level_m2 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(level_h2)  # 池化层
        flayer = GlobalAveragePooling2D()(level_m2)
        fc1 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax')(fc1)
        #compile it
        model = Model(inputs=self.model_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=categorical_crossentropy)
        print(' The cnn model with {} layers is estabished.'.format('4'))
        modelname = 'CNN_4'
        return model, modelname

    def CreateLightCNNModel(self):
        h = Conv2D(kernel_size=3, filters=32, activation='relu', strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0001))(self.model_input)
        h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)
        h = Conv2D(kernel_size=3, filters=64, activation='relu', strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0001))(h)
        h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)
        flayer = Flatten()(h)
        fc1 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax')(fc1)
        # compile it
        model = Model(inputs=self.model_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=categorical_crossentropy)
        print(' The cnn model with {} layers is estabished.'.format('2'))
        modelname = 'CNN_2'
        return model, modelname

    def CreateLSTMcounterpart(self):
        h = Reshape((10, 26), name='squeeze')(self.model_input)
        h = LSTM(128, return_sequences=False)(h)
        # flayer = Flatten()(h)
        fc1 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(h)  # 全连接层
        y_pred = Activation('softmax')(fc1)
        # compile it
        model = Model(inputs=self.model_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=categorical_crossentropy)
        print('The lstm model with 128 hidden units is estabished.')
        modelname = 'lstm_128'
        return model, modelname

    def CreateLSTMmodel(self):
        h = LSTM(128, return_sequences=True)(self.model_input)
        # flayer = Flatten()(h)
        fc1 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(h)  # 全连接层
        y_pred = Activation('softmax')(fc1)
        # compile it
        model = Model(inputs=self.model_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=categorical_crossentropy)
        print('The lstm model with 128 hidden units is estabished.')
        modelname = 'real_lstm_128'
        return model, modelname

    def CreateCLSTMmodel(self):
        h = ConvLSTM2D(filters=32, kernel_size=(3, 3)
                                    , data_format='channels_first'
                                    , recurrent_activation='hard_sigmoid'
                                    , activation='relu'
                                    , padding='same', return_sequences=True)(self.model_input)
        h = MaxPooling3D(pool_size=(1, 2, 2), padding='valid', data_format='channels_first')(h)
        flayer = TimeDistributed(Flatten())(h)
        fc1 = TimeDistributed(Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal'))(flayer)
        y_pred = Activation('softmax')(fc1)
        # compile it
        model = Model(inputs=self.model_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=categorical_crossentropy)
        print(' The convlutional lstm model is estabished.')
        modelname = 'ConvLstm'
        return model, modelname


class operation():
    def __init__(self,model,modelname,basePath,crossValType='5folds'):
        assert (crossValType in ('5folds', 'loo','nonCross5folds','nonCrossLoo','new_serieslstm','new_convlstm'))
        # if crossValType == '5folds':
        #     strr = 'A1b_binaryDataSpeech.py'
        # else:
        #     strr = 'A1c_leaveOneOutData.py'
        self.crossValType = crossValType
        if crossValType == '5folds':
            print('It is about the 5-folds cross validation method!')
            self.dataRelated = importlib.import_module('A1b_binaryDataSpeech')
        elif crossValType == 'loo':
            self.dataRelated = importlib.import_module('A1c_leaveOneOutData')
            print('It is about the leave-one-out validation method!')
        elif crossValType == 'nonCross5folds' or crossValType == 'nonCrossLoo':
            self.dataRelated = importlib.import_module('A1d_nonCrossPeople')
        elif crossValType in ('new_serieslstm', 'new_convlstm'):
            self.dataRelated = importlib.import_module('A1e_clstmStyleDataFeeding')
        else:
            assert(0)
        self.model = model
        self.clearPath = basePath
        self.basePath = os.path.join(self.clearPath,modelname)
        self.baseSavPath = []
        self.baseSavPath.append(self.basePath)
        self.baseSavPath.append(self.basePath+'_weights')
        self.modelname = modelname

    def train(self,foldsflag=0, batch_size=32,epoches=10,recorder=None):
        opit = self.dataRelated.dataset_operator()
        data = self.dataRelated.buildTrainValtest(opit.datatable, opit.labeltable)
        if self.crossValType in ('5folds','loo'):
            data.cutintopieces(flag=foldsflag)
        elif self.crossValType == 'nonCrossLoo':
            data.cutintopiecesLoo(flag=foldsflag)
        elif self.crossValType == 'nonCross5folds':
            data.cutintopieces5folds(flag=foldsflag)
        elif self.crossValType == 'new_serieslstm':
            data.cutintopiecesLoo(flag=foldsflag)
        elif self.crossValType == 'new_convlstm':
            data.cutintopiecesLoo(flag=foldsflag,reserve=True)
        else:
            assert(0)
        data.generatorSetting(batch_size=batch_size)
        if self.crossValType in ('new_serieslstm', 'new_convlstm'):
            num_data =len(data.trainList)
        else:
            num_data = sum(data.setCounting(data.trainPieces))  # 获取数据的数�?
        os.system('pkill tensorboard')
        os.system('rm -rf ./checkpoints/files_summary/* ')
        train_writter = tf.summary.FileWriter(os.path.join(os.getcwd(), 'checkpoints', 'files_summary'))
        os.system('tensorboard --logdir=/home/zhaok14/example/PycharmProjects/shardware/project1/checkpoints/files_summary/ --port=6006 &')
        print('\n')
        print(90 * '*')
        print(40 * ' ',self.modelname)
        print(39 * ' ','fold_'+str(foldsflag))
        print(90 * '*')

        iterations_per_epoch = data.iteration_per_epoch
        # iterations_per_epoch = 30
        print('trainer info:')
        print('training data size: %d' % num_data)
        print('increased epoches: ', epoches)
        print('minibatch size: %d' % batch_size)
        print('iterations per epoch: %d' % iterations_per_epoch)

        sess = k.get_session()
        train_writter.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        best_score = 0
        # epoch = 2
        duration = 0
        for i in range(0, epoches):
            iteration = 0
            yielddatas = data.data_genetator()
            pbar = tqdm(yielddatas)
            for input, labels in pbar:
                stime = time.time()
                if self.crossValType in ('new_serieslstm', 'new_convlstm'):
                    loss = self.model.train_on_batch(input, labels)
                else:
                    loss = self.model.train_on_batch(input, labels, class_weight=data.classWeights)
                dtime = time.time() - stime
                duration = duration + dtime
                train_summary = tf.Summary()
                train_summary.value.add(tag='loss', simple_value=loss)
                train_writter.add_summary(train_summary, iteration + i * iterations_per_epoch)
                pr = 'epoch:%d/%d,iteration: %d/%d ,loss: %s' % (epoches, i, iterations_per_epoch, iteration, loss)
                pbar.set_description(pr)
                if iteration == iterations_per_epoch:
                    break
                else:
                    iteration += 1
            pbar.close()
            #measure model after each epoch
            if self.crossValType in ('new_serieslstm', 'new_convlstm'):
                self.TestSeriesModel(sess=sess, type='train', data=data, data_count=-1, writer=train_writter, step=i)
                metrics = self.TestSeriesModel(sess=sess, type='test', data=data, data_count=-1, writer=train_writter, step=i)
            else:
                self.TestModel(sess=sess, type = 'train', data=data, data_count=-1, writer=train_writter, step=i)
                metrics = self.TestModel(sess=sess, type = 'test', data=data, data_count=-1, writer=train_writter, step=i)
            if i > 0:
                if metrics['score'] >= best_score:
                    self.metrics = metrics
                    self.metrics['epoch'] = i
                    best_score = metrics['score']
                    clrdir(self.clearPath)
                    self.savpath = []
                    self.savpath.append((self.baseSavPath[0] + '_epoch' + str(i) + '.h5'))
                    self.savpath.append((self.baseSavPath[1] + '_epoch' + str(i) + '.h5'))
                    self.model.save(self.savpath[0])
                    self.model.save_weights(self.savpath[1])
        if 'epoch' in self.metrics.keys():
            print('The best metric (without restriction) took place in the epoch: ', self.metrics['epoch'])
            print('Sensitivity: {}; Specificity: {}; Score: {}; Accuracy: {}'.format(self.metrics['sensitivity'],self.metrics['specificity'],self.metrics['score'],self.metrics['accuracy']))
            # self.TestGenerability(feature_type = feature_type, weightspath=self.savpath[1])
        else:
            print('The best metric (without restriction) is not found. Done!')
        print('Training duration: {}s'.format(round(duration, 2)))
        if self.crossValType == 'loo':
            print('For the loo method, recheck the test set to have it recorded..')
            assert(not(recorder is None))
            self.evaluating(recorder,type = 'test', data=data, data_count=-1)
        print()
        return (self.metrics['score'], self.metrics['accuracy'],self.metrics['sensitivity'],self.metrics['specificity'],self.metrics['precision'])

    # only for A1c_leaveOneOutData use, since the fielname is not added for
    def evaluating(self,recordingObject,data,type,data_count=32,weightsdir=None):
        assert(self.crossValType == 'loo')
        if weightsdir is None:
            weightsdir = os.path.join(os.getcwd(),'modelfiles')
        WeightsFile = [i for i in os.listdir(weightsdir) if 'weights' in i]
        assert (len(WeightsFile) == 1)
        self.model.load_weights(os.path.join(weightsdir, WeightsFile[0]))
        for layer in self.model.layers:
            layer.trainable = False
        assert (type in ('train', 'test'))
        if type == 'test':
            num_data = len(data.testList)  # 获取数据的数量
        else:
            num_data = len(data.trainList)
        if (data_count <= 0 or data_count > num_data):  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data
        try:
            ran_num = randint(0, num_data - 1)  # 获取一个随机数
            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0
            overall_fp = 0
            start = time.time()
            # maplist = {0: 'Present', 1: 'Absent', 2:'Weak', 3:'Uncertain'}
            maplist = {0: 'Present', 1: 'Absent'}
            # data_count = 200
            recordingObject.recording[data.looTestMark] = []
            for i in tqdm(range(data_count)):
                data_input, data_labels, timestamp, ori_label = data.getNonRepetitiveData((ran_num + i) % num_data,
                                                                    type=type,mark=True)  # 从随机数开始连续向后取一定数量数据
                data_pre = self.model.predict_on_batch(np.expand_dims(data_input, axis=0))
                predictions = np.argmax(data_pre[0], axis=0)
                if predictions != data_labels:
                    recordingObject.recording[data.looTestMark].append(tuple(list(timestamp) + [predictions] + [round(np.max(data_pre[0]),4)] + [ori_label] ))
                tp, fp, tn, fn = Compare4(predictions, data_labels)  # 计算metrics
                overall_p += tp + fn
                overall_n += tn + fp
                overall_tp += tp
                overall_tn += tn
                overall_fp += fp
            if overall_p != 0:
                sensitivity = overall_tp / overall_p * 100
                sensitivity = round(sensitivity, 2)
            else:
                sensitivity = 'None'
            if overall_n != 0:
                specificity = overall_tn / overall_n * 100
                specificity = round(specificity, 2)
            else:
                specificity = 'None'
            if sensitivity != 'None' and specificity != 'None':
                score = (sensitivity + specificity) / 2
                score = round(score, 2)
            else:
                score = 'None'
            if (overall_tp + overall_fp) != 0:
                precision = overall_tp / (overall_tp + overall_fp) * 100
                precision = round(precision,2)
            else:
                precision = 0
            accuracy = (overall_tp + overall_tn) / (overall_p + overall_n) * 100
            accuracy = round(accuracy, 2)
            end = time.time()
            dtime = round(end - start, 2)
            strg = '*[测试结果] 片段识别 {0} 敏感度：{1}%, 特异度： {2}%, 精度： {3}%, 得分： {4}, 准确度： {5}%, 用时: {6}s.'.format(type, sensitivity,specificity, precision, score,accuracy, dtime)
            tqdm.write(strg)
            metrics = {'data_set': type, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'score': score, 'accuracy': accuracy}
            return metrics

        except StopIteration:
            print('*[Error] Model Test Error. please check data format.')

    def TestSeriesModel(self, sess, data, type, writer, data_count=32, show_ratio=True, step=0):
        '''
                测试检验模型效果
                '''
        assert (type in ('train', 'test'))
        if type == 'test':
            num_data = len(data.testList)  # 获取数据的数量
        else:
            num_data = len(data.trainList)
        if (data_count <= 0 or data_count > num_data):  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data
        try:
            ran_num = randint(0, num_data - 1)  # 获取一个随机数
            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0
            start = time.time()
            cm_pre = []
            cm_lab = []
            # maplist = {0: 'Present', 1: 'Absent', 2:'Weak', 3:'Uncertain'}
            maplist = {0: 'Present', 1: 'Absent'}
            # data_count = 200
            for i in tqdm(range(data_count)):
                data_input, data_labels = data.getNonRepetitiveData((ran_num + i) % num_data, type=type)  # 从随机数开始连续向后取一定数量数据
                data_pre = self.model.predict_on_batch(np.expand_dims(data_input, axis=0))
                predictions = np.argmax(data_pre[0], axis=1)
                cm_pre = cm_pre + [maplist[i] for i in predictions]
                cm_lab = cm_lab + [maplist[i] for i in data_labels]
                for prediction,label in zip(predictions,data_labels):
                    tp, fp, tn, fn = Compare4(prediction, label)  # 计算metrics
                    overall_p += tp + fn
                    overall_n += tn + fp
                    overall_tp += tp
                    overall_tn += tn
            if overall_p != 0:
                sensitivity = overall_tp / overall_p * 100
                sensitivity = round(sensitivity, 2)
            else:
                sensitivity = 'None'
            if overall_n != 0:
                specificity = overall_tn / overall_n * 100
                specificity = round(specificity, 2)
            else:
                specificity = 'None'
            if sensitivity != 'None' and specificity != 'None':
                score = (sensitivity + specificity) / 2
                score = round(score, 2)
            else:
                score = 'None'
            accuracy = (overall_tp + overall_tn) / (overall_p + overall_n) * 100
            accuracy = round(accuracy, 2)
            end = time.time()
            dtime = round(end - start, 2)
            strg = '*[测试结果] 片段识别 {0} 敏感度：{1}%, 特异度： {2}%, 得分： {3}, 准确度： {4}%, 用时: {5}s.'.format(type, sensitivity,
                                                                                                specificity, score,
                                                                                                accuracy, dtime)
            tqdm.write(strg)

            assert (len(cm_lab) == len(cm_pre))
            img_cm = plot_confusion_matrix(cm_lab, cm_pre, list(maplist.values()), tensor_name='MyFigure/cm',
                                           normalize=False)
            writer.add_summary(img_cm, global_step=step)
            summary = tf.Summary()
            summary.value.add(tag=type + '/sensitivity', simple_value=sensitivity)
            summary.value.add(tag=type + '/specificity', simple_value=specificity)
            summary.value.add(tag=type + '/score', simple_value=score)
            summary.value.add(tag=type + '/accuracy', simple_value=accuracy)
            writer.add_summary(summary, global_step=step)

            metrics = {'data_set': type, 'sensitivity': sensitivity, 'specificity': specificity, 'score': score,
                       'accuracy': accuracy}
            return metrics
        except StopIteration:
            print('*[Error] Model Test Error. please check data format.')

    def TestModel(self, sess, data, type, writer, data_count=32, show_ratio=True, step=0):
        '''
        测试检验模型效果
        '''
        assert(type in ('train','test'))
        if type == 'test':
            num_data = len(data.testList)  # 获取数据的数量
        else:
            num_data = len(data.trainList)
        if (data_count <= 0 or data_count > num_data):  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data
        try:
            ran_num = randint(0, num_data - 1)  # 获取一个随机数
            overall_p = 0
            overall_n = 0
            overall_tp = 0
            overall_tn = 0
            overall_fp = 0
            start = time.time()
            cm_pre = []
            cm_lab = []
            # maplist = {0: 'Present', 1: 'Absent', 2:'Weak', 3:'Uncertain'}
            maplist = {0: 'Present', 1: 'Absent'}
            # data_count = 200
            for i in tqdm(range(data_count)):
                data_input, data_labels = data.getNonRepetitiveData((ran_num + i) % num_data,type=type)  # 从随机数开始连续向后取一定数量数据
                data_pre = self.model.predict_on_batch(np.expand_dims(data_input, axis=0))
                predictions = np.argmax(data_pre[0], axis=0)
                cm_pre.append(maplist[predictions])
                cm_lab.append(maplist[data_labels])
                tp, fp, tn, fn = Compare4(predictions, data_labels)  # 计算metrics
                overall_p += tp + fn
                overall_n += tn + fp
                overall_tp += tp
                overall_tn += tn
                overall_fp += fp
            if overall_p != 0:
                sensitivity = overall_tp / overall_p * 100
                sensitivity = round(sensitivity, 2)
            else:
                sensitivity = 'None'
            if overall_n != 0:
                specificity = overall_tn / overall_n * 100
                specificity = round(specificity, 2)
            else:
                specificity = 'None'
            if sensitivity != 'None' and specificity != 'None':
                score = (sensitivity + specificity) / 2
                score = round(score, 2)
            else:
                score = 'None'
            if (overall_tp + overall_fp) != 0:
                precision = overall_tp / (overall_tp + overall_fp) * 100
                precision = round(precision,2)
            else:
                precision = 0
            accuracy = (overall_tp + overall_tn) / (overall_p + overall_n) * 100
            accuracy = round(accuracy, 2)
            end = time.time()
            dtime = round(end - start, 2)
            strg = '*[测试结果] 片段识别 {0} 敏感度：{1}%, 特异度： {2}%, 精度： {3}%, 得分： {4}, 准确度： {5}%, 用时: {6}s.'.format(type, sensitivity, specificity, precision, score, accuracy, dtime)
            tqdm.write(strg)

            assert (len(cm_lab) == len(cm_pre))
            img_cm = plot_confusion_matrix(cm_lab, cm_pre, list(maplist.values()),tensor_name='MyFigure/cm', normalize=False)
            writer.add_summary(img_cm, global_step=step)
            summary = tf.Summary()
            summary.value.add(tag=type + '/sensitivity', simple_value=sensitivity)
            summary.value.add(tag=type + '/precision', simple_value=precision)
            summary.value.add(tag=type + '/specificity', simple_value=specificity)
            summary.value.add(tag=type + '/score', simple_value=score)
            summary.value.add(tag=type + '/accuracy', simple_value=accuracy)
            writer.add_summary(summary, global_step=step)

            metrics = {'data_set': type, 'sensitivity': sensitivity, 'specificity': specificity, "precision":precision, 'score': score,'accuracy': accuracy}
            return metrics

        except StopIteration:
            print('*[Error] Model Test Error. please check data format.')

    def statisticfeatures(self,seriesdata):
        '''

        :param seriesdata: a list with tuples as elements. Moreover, the elements in the tuple represent different metrics.
        :return: two lists. One is for mean values, the other is for standard variations.

        '''
        so = zip(*seriesdata)
        resultMean = []
        resultStd = []
        for x in so:
            mean = round(float(np.mean(x)), 2)
            std = round(float(np.std(x)), 2)
            resultMean.append(mean)
            resultStd.append(std)
        return resultMean, resultStd

    def LoopValidation(self,batch_size=16,epoches=40,wcpath=None):
        if self.crossValType in ('5folds','nonCross5folds'):
            metric5 = []
            for i in range(5):
                metric5.append(self.train(foldsflag=i, batch_size=batch_size, epoches=epoches))
            means,stds = self.statisticfeatures(metric5)
            print()
            print()
            if self.crossValType == '5folds':
                print('the results of the 5-folds cross validation method:')
            else:
                print('the results of the non-cross-peopled 5-folds cross validation method:')
            print('score: {}+/-{}'.format(means[0], stds[0]))
            print('accuracy: {}+/-{}'.format(means[1], stds[1]))
            print('sensitivity: {}+/-{}'.format(means[2], stds[2]))
            print('specificity: {}+/-{}'.format(means[3], stds[3]))
            print('precision: {}+/-{}'.format(means[4], stds[4]))
            print()
            print()
        elif self.crossValType in ('loo', 'new_serieslstm', 'new_convlstm'):
            if wcpath is None:
                wcpath = os.path.join(os.getcwd(), 'data')
            wc = len(os.listdir(wcpath))
            metric22 = []
            recorder = misclassifiedManagement()
            # wc = 1
            for i in range(wc):
                metric22.append(self.train(foldsflag=i, batch_size=batch_size, epoches=epoches,recorder=recorder))
            recorder.alright()
            import pickle
            fp = os.path.join( os.getcwd(),'metrics.pickle')
            pon = open(fp,'wb')
            pickle.dump(metric22,pon)
            pon.close()
            means,stds = self.statisticfeatures(metric22)
            print()
            print()
            print('the results of the leave-one-out cross validation method:')
            print('score: {}+/-{}'.format(means[0], stds[0]))
            print('accuracy: {}+/-{}'.format(means[1], stds[1]))
            print('sensitivity: {}+/-{}'.format(means[2], stds[2]))
            print('specificity: {}+/-{}'.format(means[3], stds[3]))
            print('precision: {}+/-{}'.format(means[4], stds[4]))
            print()
            print()
        elif self.crossValType == 'nonCrossLoo':
            metric60 = []
            for i in range(60):
                metric60.append(self.train(foldsflag=i, batch_size=batch_size, epoches=epoches))
            means, stds = self.statisticfeatures(metric60)
            print()
            print()
            print('the results of the non-cross-peopled leave-one-out validation method:')
            print('score: {}+/-{}'.format(means[0], stds[0]))
            print('accuracy: {}+/-{}'.format(means[1], stds[1]))
            print('sensitivity: {}+/-{}'.format(means[2], stds[2]))
            print('specificity: {}+/-{}'.format(means[3], stds[3]))
            print('precision: {}+/-{}'.format(means[4], stds[4]))
            print()
            print()
        else:
            assert(0)
        gc.collect()



if __name__ == '__main__':
    a = Networks((10,26,1))
    lcnn,_ = a.CreateLightCNNModel()
    lcnn.summary()
    lstmc, _ = a.CreateLSTMcounterpart()
    lstmc.summary()