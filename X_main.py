from A2b_binaryNetworks import Networks, operation
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only display error and warning; for 1: all info; for 3: only error.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显�? 按需分配
set_session(tf.Session(config=config))
import sys
class logger(object):
    def __init__(self,filename=None):
        if filename is None:
            filename = os.path.join(os.getcwd(), 'Default.log')
        self.terminal = sys.stdout
        self.log = open(filename,'w')
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

if __name__ == '__main__':
    sys.stdout = logger(filename=os.path.join(os.getcwd(), 'label2profiling.log'))
    nn = Networks(inputShape=(10,26,1))
    # model, modelname = nn.CreateRegularCNNModel()
    model, modelname = nn.CreateLightCNNModel()
    ###
    # oper = operation(modelname=modelname,model=model,basePath=os.path.join(os.getcwd(),'modelfiles',),crossValType='nonCross5folds')
    # oper.LoopValidation(epoches=40)
    # ###
    # oper = operation(modelname=modelname, model=model, basePath=os.path.join(os.getcwd(), 'modelfiles', ),crossValType='nonCrossLoo')
    # oper.LoopValidation(epoches=40)
    # ###
    # oper = operation(modelname=modelname, model=model, basePath=os.path.join(os.getcwd(), 'modelfiles', ),crossValType='5folds')
    # oper.LoopValidation(epoches=40)
    ###
    # model, modelname = nn.CreateLSTMcounterpart()
    # nn = Networks(inputShape=(60,260))
    # nn = Networks(inputShape=(60, 1, 10, 26))
    # model, modelname = nn.CreateLSTMmodel()
    # model, modelname = nn.CreateCLSTMmodel()
    oper = operation(modelname=modelname, model=model, basePath=os.path.join(os.getcwd(), 'modelfiles'),crossValType='loo')
    oper.LoopValidation(batch_size=128,epoches=40)
