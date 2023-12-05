from numpy import *
import tensorflow as tf
from PIL import Image
import os
from matplotlib import pyplot as plt

#paths
path1 = r'##############'    #images path    
path2 = r'##############'   #output label path
model_path = r'#########' #model storage path

#Global train and test lists
train_x = []
train_y = []
test_x = []
test_y = []

#parameters
n_hidden_1 = 6400
n_hidden_2 = 6400
n_hidden_3 = 6400
n_input = 6400
n_output = 6400
batch_size = 2048
grid_size = 80
total_epoch = 4000
learning_rate = 0.00005
h,w = 0,0
num_imgs = 244*4

#tf input
x = tf.placeholder(tf.float32, shape=[None,6400])  #6400 flat one channel input, say 'blue-ch'
y = tf.placeholder(tf.float32,shape=[None,6400])     #6400 flat output label mask containing only '0' or '1'

#create image data model
def get_data():
    print('Reading Data for Training')
    #imlist = os.listdir(path1)
    #masklist = os.listdir(path2)
    #print (imlist)
    #print (masklist)
    for i in range(num_imgs):
        img = Image.open('/home/ubuntu/Pictures/less_train/'+str(i+1)+'.jpg')
        #img_g = array(img)[:,:,1]
        #plt.imshow(img_g)
        #plt.show()
        #mask = Image.open(os.path.join(path2,masklist[i]))
        mask = Image.open('/home/ubuntu/Pictures/less_train/'+'mask_new'+str(i+1)+'.bmp')
        #print (unique(array(mask)))
        #plt.imshow(img)
        #plt.show()
        #plt.imshow(mask)
        #plt.show()
        l_ht,l_wt = (array(mask).shape)
        l_ht = (int)(l_ht/grid_size)
        l_wt = (int)(l_wt/grid_size)
        #print (i,l_ht,l_wt,img.size,mask.size)
        if (i<10000):
            for j in range(l_ht):
                for k in range(l_wt):
                    img_new = img.crop((k*grid_size,j*grid_size,(k+1)*grid_size,(j+1)*grid_size))   #cropin grid of size 80x80 for in_image
                    mask_new = mask.crop((k*grid_size,j*grid_size,(k+1)*grid_size,(j+1)*grid_size)) #cropin grid of size 80x80 for label image
                    mask_01 = array(mask_new)/255 #converting mask to 0 and 1
                    mask_01 = mask_01.astype(int)
                    #print(unique(mask_01))
                    #print(unique(img_new))
                    train_x.append(array(reshape(array(img_new)[:,:,2:3],(grid_size,grid_size))).flatten())    #taking only one channel
                    train_y.append(array(mask_01).flatten())
                    #print (array(train_y).shape)
                    '''if(i==20):
                        test_x.append(array(img_new))
                        test_y.append(array(mask_new))'''
                    #plt.imshow(img_new)
                    #plt.show()
                    #plt.imshow(mask_new)
                    #plt.show()
            #print (array(train_y).shape)
        else:
             for j in range(l_ht):
                for k in range(l_wt):
                    img_new = img.crop((k*grid_size,j*grid_size,(k+1)*grid_size,(j+1)*grid_size))   #cropin grid of size 80x80 for in_image
                    mask_new = mask.crop((k*grid_size,j*grid_size,(k+1)*grid_size,(j+1)*grid_size)) #cropin grid of size 80x80 for label image
                    mask_01 = array(mask_new)/255 #converting mask to 0 and 1
                    mask_01 = mask_01.astype(int)
                    #print(unique(mask_01))
                    #print(unique(img_new))
                    test_x.append(array(reshape(array(img_new)[:,:,2:3],(grid_size,grid_size))).flatten())    #taking only one channel
                    test_y.append(array(mask_01).flatten())          
    print('Finished Reading Data')
    print (array(train_x).shape)
    print (array(train_y).shape)
    #print (train_x)
    #print (train_y)
    print (array(test_x).shape)
    print (array(test_y).shape)    
    #print (test_x)
    #print (test_y)
    
#fetch data
get_data()

#layers weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=5.0)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.5)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.05)),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_output], stddev=0.001))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=5.0)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.5)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], stddev=0.05)),
    'out': tf.Variable(tf.random_normal([n_output], stddev=0.001))
}


#create model
def perceptron(data,weights,biases):
    layer_1 = tf.add(tf.matmul(data, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)    
    output_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    return output_layer


#construct model
pred = perceptron(x,weights,biases)

#loss and optimizer
loss = tf.nn.l2_loss(pred-y) 
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# 'saver' to save and restore all the variables
saver = tf.train.Saver()

#Running session
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Training cycle
    for epoch in range(total_epoch):
        epoch_loss = 0
        i=0
        # Loop over all batches
        while (i<len(train_x)):
            start = i
            end = i + batch_size
            batch_x = array(train_x[start:end])
            batch_y = array(train_y[start:end])
            _,c = sess.run([optimizer,loss], feed_dict ={x: batch_x, y: batch_y})
            #epoch_loss = c
            epoch_loss += c
            i += batch_size
        print('Epoch', epoch+1,'completed out of ', total_epoch,'loss:',epoch_loss)
        #print('Loss_test:', loss.eval({x: test_x, y:test_y}, session=sess))

    save_path = saver.save(sess, model_path)
    print ("Model saved in file: %s" % save_path)
