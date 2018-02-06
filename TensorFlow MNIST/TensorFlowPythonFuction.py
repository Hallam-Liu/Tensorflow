#!/usr/bin/python
# -*- coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import tensorflow.python.platform
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
#----------TensorFlow extract MNIST Data file------------Liuhr
# if not, download the file and save to the filepath, return the file path
def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
 #---check local whether there is work_directory
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.','the filepath is',filepath)
  return filepath
#---------------------------------------------------------------------------
# to read a binary file
def _read32(bytestream):
#，作用是从文件流中动态读取4位数据并转换为uint32的数据。
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

#---------------------------------------------------------------------------
#image文件的前四位为魔术码（magic number），
#只有检测到这4位数据的值和2051相等时，才代表这是正确的image文件，才会继续往下读取
# 接下来继续读取之后的4位，代表着image文件中，所包含的图片的数量（num_images）。
# 再接着读4位，为每一幅图片的行数（rows），再后4位，为每一幅图片的列数（cols）。
# 最后再读接下来的rows * cols * num_images位，即为所有图片的像素值。
# 最后再将读取到的所有像素值装换为[index, rows, cols, depth]的4D矩阵。这样就将全部的image数据读取了出来。
#-------------------------------------------------------------------------------------------
def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data
#-------------------------------------------------------
# 同理，对于MNIST的labels文件：
# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
# offset	type	value	description
# 0000	32 bit integer	0x00000801(2051)	magic number
# 0004	32 bit integer	60000	number of items
# 0008	unsigned byte	??	label
# 0009	unsigned byte	??	label
# ......	 	 	 
# xxxx	unsigned byte	??	label
#-------------------------------------------------------
#同样的也是依次读取文件的魔术码以及标签总数，最后把所有图片的标签读取出来，成一个长度为num_items的1D的向量。
#-------------------------------------------------------

# 正如文章开头提到one_hot的作用，这里将1D向量中的每一个值，编码成一个长度为num_classes的向量，向量中对应于该值的位置为1，
# 其余为0，所以one_hot将长度为num_labels的向量编码为一个[num_labels, num_classes]的2D矩阵。
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels

# import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if fake_data:
    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
    data_sets.train = fake()
    data_sets.validation = fake()
    data_sets.test = fake()
    return data_sets
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000
  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)
  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)
  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)
  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)
  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]
  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, dtype=dtype)
  return data_sets
import tensorflow as tf