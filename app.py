import tensorflow as tf
import random
import shutil
import glob
import os

os.chdir('data/Santi')
if os.path.isdir('train/santi') is False:
    os.makedirs('train/santi')
    os.makedirs('valid/santi')
    os.makedirs('test/santi')

for c in random.sample(glob.glob('santi*'), 234):
    shutil.move(c, 'train/santi')
for c in random.sample(glob.glob('santi*'), 29):
    shutil.move(c, 'valid/santi')
for c in random.sample(glob.glob('santi*'), 29):
    shutil.move(c, 'test/santi')
