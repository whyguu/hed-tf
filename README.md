# Train HED on Your Own Data

### training data
1. put training data and labels in ./data/dataset/train_data/
2. generate train.txt like the already existed one

### vgg16 weights
google for vgg16.npy weights file and put it in ./data/weights/initial_weights/

no need to google, it is here https://mega.nz/#!VqAEETba!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM

p.s. if you don't use vgg16 weights just comment 'hed_class.assign_init_weights(sess)' in train.py

### train
cd to the root directory './hed-tf'

python train.py -gpu '0' # default gpu 0

p.s. it seems that if you do not have gpu ,tf will run it in cpu in this program

## Attention
change the learning rate in train.py

this repository is not exactly identical to hed paper.

I use dilate conv at conv1_1, you can change it easily.

for other changes , to see the paper and official code carefully.
