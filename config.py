import sys
import warnings
from sklearn.preprocessing import StandardScaler

class DefaultConfig(object):
    seed = 666
    device = 7

    data_path = 'data'
    save_path = 'log'
    scaler = StandardScaler()
    
    batch_size = 6
    lr = 1e-5
    
    weight_decay = 1e-2

    step_size = 100
    gamma = 0.3
    
    epochs = 500

    def parse(self, kwargs):
        '''
        customize configuration by input in terminal
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has no attribute %s' % k)
            setattr(self, k, v)

    def output(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


class Logger(object):
    def __init__(self, file_name='Default.log'):

        self.terminal = sys.stdout
        self.log = open(file_name, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass