from numpy import *
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import time

start_time = time.time()
train_x = []
train_y = []
test_x = []
test_y = []
pix_grid = 6400
n_hl1 = pix_grid
n_hl2 = pix_grid
n_hl3 = pix_grid
batch_size = 2048
inp_nodes = pix_grid
op_nodes = pix_grid

x = tf.placeholder(tf.float32, shape=[None,pix_grid])  #6400 flat one channel input, say 'blue-ch'
y = tf.placeholder(tf.float32,shape=[None,pix_grid])     #6400 flat output label mask containing only '0' or '1'

#path1 = r'/home/ubuntu/test_imgs/orig'    #path of folder of images    
#path2 = r'/home/ubuntu/test_imgs/mask'  #path of folder of masks
model_path = r'/home/ubuntu/models/rust_378x4imgs_3rdch_3lay_g80.ckpt'
#path3 = r'D:\rust\rust_metal_segmentation\M15.jpg'
#path4 = r''
#x_1,y_1 = 240,320   #dimensions of input image
grid_size = 80

######################
#x = tf.placeholder(tf.float32, shape=[None,pix_grid])
#y = tf.placeholder(tf.float32, shape=[None,pix_grid])

test_x = []
test_y = []
idx = 8 #index of test img
#imlist = os.listdir(path1)
#masklist = os.listdir(path2)
#t_x = imlist[13]
#t_y = masklist[13]
#print (t_x)
#print (t_y)
l_ht,l_wt = 0,0
#orig_mask_whole = Image.open('/home/ubuntu/Notebooks/comb_imgs/'+'OB ('+str(idx)+').jpg')
orig_mask_whole = Image.open('/home/ubuntu/Pictures/test/'+'test'+str(idx)+'.jpg')
orig_mask,a,b = orig_mask_whole.split()
#plt.imshow(orig_mask)
#plt.show()
for i in range(1):
        #img = Image.open('/home/ubuntu/Notebooks/comb_imgs/'+'OB ('+str(idx)+').jpg')
        img = Image.open('/home/ubuntu/Pictures/test/'+'test'+str(idx)+'.jpg')
	#mask = Image.open('/home/ubuntu/Notebooks/comb_imgs/'+'mask_new'+str(idx)+'.bmp')
        l_ht,l_wt,ch = (array(img).shape)
        l_ht = (int)(l_ht/grid_size)
        l_wt = (int)(l_wt/grid_size)
        for j in range(l_ht):
            for k in range(l_wt):
                    img_new = img.crop((k*grid_size,j*grid_size,(k+1)*grid_size,(j+1)*grid_size))   #cropin grid of size 80x80 for in_image
                    #mask_new = mask.crop((k*grid_size,j*grid_size,(k+1)*grid_size,(j+1)*grid_size)) #cropin grid of size 80x80 for label image
                    #mask_01 = array(mask_new)/255 #converting mask to 0 and 1
                    #mask_01 = mask_01.astype(int)
                    #print(unique(mask_01))
                    #print(unique(img_new))
                    test_x.append(array(reshape(array(img_new)[:,:,2:3],(grid_size,grid_size))).flatten())    #taking only one channel
                    #test_y.append(array(mask_01).flatten())

print (array(test_x).shape)
print (array(test_y).shape)  
print (l_ht)
print (l_wt)

def neural_network(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([inp_nodes, n_hl1],stddev=5.0)),
	                   'biases':tf.Variable(tf.random_normal([n_hl1],stddev=5.0))}

    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_hl1, n_hl2],stddev=0.5)),
    	                   'biases':tf.Variable(tf.random_normal([n_hl2],stddev=0.5))}

    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_hl2, n_hl3],stddev=0.05)),
                          'biases':tf.Variable(tf.random_normal([n_hl3],stddev=0.05))}

    #hidden_layer_4 = {'weights':tf.Variable(tf.random_normal([n_hl3, n_hl4],stddev=0.01)),
    #	                   'biases':tf.Variable(tf.random_normal([n_hl4],stddev=0.01))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_hl3, op_nodes],stddev=0.001)),
                         'biases':tf.Variable(tf.random_normal([op_nodes],stddev=0.001))}

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.sigmoid(l1)
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.sigmoid(l2)
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.sigmoid(l3)
    #l4 = tf.add(tf.matmul(l3, hidden_layer_4['weights']), hidden_layer_4['biases'])
    #l4 = tf.nn.sigmoid(l4)
   
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    #output = tf.nn.sigmoid(output)
    #output = tf.round(output)
    
    #output = tf.Print(output,[output])
    return output

prediction = neural_network(x)
#prediction = tf.cast(tf.round(prediction),tf.float32) 
loss = tf.nn.l2_loss(prediction-y)
#loss = tf.losses.huber_loss(y,prediction,delta=0.5) 
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

#########seperate test images load################   
'''test_x = []
img = Image.open(path3)
plt.imshow(img)
plt.show()
l_ht,l_wt,c = (array(img).shape)
l_ht = (int)(l_ht/40)
l_wt = (int)(l_wt/40)
for j in range(l_ht):
    for k in range(l_wt):
                    img_new = img.crop((k*40,j*40,(k+1)*40,(j+1)*40))   #cropin grid of size 80x80 for in_image
                    #mask_new = mask.crop((k*40,j*40,(k+1)*40,(j+1)*40)) #cropin grid of size 80x80 for label image
                    #mask_01 = array(mask_new)/255 #converting mask to 0 and 1
                    #mask_01 = mask_01.astype(int)
                    #print(unique(mask_01))
                    #print(unique(img_new))
                    test_x.append(array(reshape(array(img_new)[:,:,0:1],(40,40))).flatten())    #taking only one channel
                    #test_y.append(array(mask_01).flatten())'''
##################################################
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print("model restored from file:")
    output = sess.run(prediction, feed_dict={x: test_x})
    print (output)
    for i in range(len(output)):
        for j in range(pix_grid):
            if(abs(output[i][j]-0) < abs(output[i][j]-1)):
                output[i][j]=0
            else:
                output[i][j]=1
    output = output.astype(uint8)
    output = array(output)*255
    print (output)
    print (array(output).shape)
    n_array = zeros((l_ht*grid_size,l_wt*grid_size))
    #r,c = 0,0
    #orig_mask = array(orig_mask)
    for x in range(l_ht):
        for y in range(l_wt):
            n_array[x*grid_size:x*grid_size+grid_size,y*grid_size:y*grid_size+grid_size] = reshape(output[x*l_wt+y],(grid_size,grid_size)).astype(uint8)
    pred_y1 = n_array.copy()
    pred_y1 = pred_y1.astype(uint8)
    #plt.imshow(n_array)
    #plt.show()
    save_img = Image.fromarray(pred_y1)
    #save_img.save('r'+str(idx)+'_g'+'_154x4imgs_3rdch_3lay_g80_new'+'.bmp')
    save_img.save('test_r'+str(idx)+'_g'+'_rust_378x4imgs_3rdch_3lay_g80'+'.bmp')
