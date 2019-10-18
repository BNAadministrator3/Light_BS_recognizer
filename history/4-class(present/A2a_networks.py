from keras import optimizers
import tensorflow as tf
from keras.models import Model
from keras.losses import categorical_crossentropy
import keras.backend as k
import os
from tqdm import tqdm
from random import randint
import time

from A1a_accessjson import dataset_operator,  buildTrainValtest, CLASS_NUM
from help_func.utilities import *
from help_func.evaluation import Compare4, plot_confusion_matrix

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
        print(' The cnn model with {} layers are estabished.'.format('4'))
        modelname = 'CNN_4'
        return model, modelname

    def CreateLightCNNModel(self):
        h = Conv2D(kernel_size=3, filters=32, activation='relu', strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(self.model_input)
        h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)
        h = Conv2D(kernel_size=3, filters=64, activation='relu', strides=1, kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=regularizers.l2(0.0005))(h)
        h = MaxPooling2D(pool_size=2, strides=None, padding="valid")(h)
        flayer = Flatten()(h)
        fc1 = Dense(CLASS_NUM, use_bias=True, kernel_initializer='he_normal')(flayer)  # 全连接层
        y_pred = Activation('softmax')(fc1)
        # compile it
        model = Model(inputs=self.model_input, outputs=y_pred)
        optimizer = optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss=categorical_crossentropy)
        print(' The cnn model with {} layers are estabished.'.format('2'))
        modelname = 'CNN_2'
        return model, modelname

class operation():
    def __init__(self,model,modelname,basePath):
        self.model = model
        self.clearPath = basePath
        self.basePath = os.path.join(self.clearPath,modelname)
        self.baseSavPath = []
        self.baseSavPath.append(self.basePath)
        self.baseSavPath.append(self.basePath+'_weights')
        self.modelname = modelname

    def train(self,foldsflag=0, batch_size=32,epoches=10):
        opit = dataset_operator()
        data = buildTrainValtest(opit.datatable, opit.labeltable)
        data.cutintopieces(flag=foldsflag)
        data.generatorSetting(batch_size=batch_size)
        num_data = sum(data.setCounting(data.trainPieces))  # 获取数据的数�?
        os.system('pkill tensorboard')
        os.system('rm -rf ./checkpoints/files_summary/* ')
        train_writter = tf.summary.FileWriter(os.path.join(os.getcwd(), 'checkpoints', 'files_summary'))
        os.system('tensorboard --logdir=/home/zhaok14/example/PycharmProjects/shardware/project1/checkpoints/files_summary/ --port=6006 &')
        print('\n')
        print(90 * '*')
        print(40 * ' ',self.modelname)
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
        return self.metrics['accuracy']

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
            start = time.time()
            cm_pre = []
            cm_lab = []
            maplist = {0: 'Present', 1: 'Absent', 2:'Weak', 3:'Uncertain'}
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
            strg = '*[测试结果] 片段识别 {0} 敏感度：{1}%, 特异度： {2}%, 得分： {3}, 准确度： {4}%, 用时: {5}s.'.format(type,sensitivity,specificity, score,accuracy, dtime)
            tqdm.write(strg)

            assert (len(cm_lab) == len(cm_pre))
            img_cm = plot_confusion_matrix(cm_lab, cm_pre, list(maplist.values()),tensor_name='MyFigure/cm', normalize=False)
            writer.add_summary(img_cm, global_step=step)
            summary = tf.Summary()
            summary.value.add(tag=type + '/sensitivity', simple_value=sensitivity)
            summary.value.add(tag=type + '/specificity', simple_value=specificity)
            summary.value.add(tag=type + '/score', simple_value=score)
            summary.value.add(tag=type + '/accuracy', simple_value=accuracy)
            writer.add_summary(summary, global_step=step)

            metrics = {'data_set': type, 'sensitivity': sensitivity, 'specificity': specificity, 'score': score,'accuracy': accuracy}
            return metrics

        except StopIteration:
            print('*[Error] Model Test Error. please check data format.')

