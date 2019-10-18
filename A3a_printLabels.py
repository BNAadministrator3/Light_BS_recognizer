import os

class misclassifiedManagement():
    '''
    Input: filename, segments, machine labels
    Outputs: append these things to a text
    mechinsm: by maintain a dictionary
    '''
    def __init__(self,filepath=None):
        if filepath is None:
            filepath = os.path.join(os.getcwd(),'wrongprofiling.log')
        self.filepath = filepath
        self.recording = {}

    def writing(self,filename,items):
        '''
        To write the misclassified items to the self.recordings
        :param filename: string data structure.
        :param items: format [ (begining,ending,tags),(),(),... ]
        :return: nothing
        '''
        self.recording[filename] = tuple(items)

    def alright(self):
        '''
        write the dictionary structure in a decent way
        :return: nothing
        '''
        with open(self.filepath,'w+') as f:
            for i in self.recording.keys():
                f.write(i)
                f.write('\n')
                for j in self.recording[i]:
                    assert(j[1]-j[0]==1)
                    temp = map(str,j)
                    aline = '  '.join(temp)
                    f.write(aline)
                    f.write('\n')


if __name__ == '__main__':
    #test
    a = misclassifiedManagement(filepath = os.path.join(os.getcwd(),'testprofiling.log'))
    a.writing('sdsdsdsd.wav',[(1,2,'Present',0.5666,3),(56,57,'Absent',0.9795,0),(34,35,'Others',0.8642,2)])
    a.writing('sdsdsdssdsdsdd.wav', [(1, 2, 'Present',0.1222,3), (56, 57, 'Absent',0.1458,4), (34, 35, 'Others',0.7642,0)])
    a.alright()
    b=1