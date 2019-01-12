# *-* coding:utf-8 *-*
'''
@author: ioiogoo
@date: 2018/1/31 19:28
'''

from data_processing import *

_,_,words, _ = data_processing()
print(len(words))
y_vec = np.zeros(
    shape=(1, len(words) ),
    dtype=np.bool
)
print(y_vec[0,5994])