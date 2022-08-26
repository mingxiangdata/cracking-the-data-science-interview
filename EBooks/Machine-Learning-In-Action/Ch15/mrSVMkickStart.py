'''
Created on Feb 27, 2011

@author: Peter
'''

from mrjob.protocol import JSONProtocol
from numpy import *

with open('kickStart2.txt', 'w') as fw:
    for _ in [1]:
        for _ in range(100):
            fw.write('["x", %d]\n' % random.randint(200))