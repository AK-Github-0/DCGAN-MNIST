
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/Users/student/Documents/OEL'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,ReLU,Reshape
import tensorflow as tf


train_data = pd.read_csv('fashion-mnist_train.csv')
train_data.head()



X_train = train_data.drop('label',axis=1)
X_train.head()

# # **Data Visualization**

fig,axe=plt.subplots(2,2)
idx = 0
for i in range(2):
    for j in range(2):
        axe[i,j].imshow(X_train[idx].reshape(28,28),cmap='gray')
        idx+=1

X_train =  X_train.astype('float32')


# The pixel data ranges from 0 to 255 hence dividing each pixel by 255,i.e,normalizing the data such that the range is within 0 to 1.
X_train = X_train/255
X_train = X_train*2 - 1.


# **NOTE**  after normalizing it is multiplied with 2 and substracted from 1 such that it ranges from (-1,1) because in DCGANs the last layer generative model activation is tanh which range is (-1,1) unlike sigmoid ranging (0,1) .

# In[10]:


print(X_train.max(),X_train.min())


# # **Simple GAN Model**

# **Generative part**


generator = Sequential()
generator.add(Dense(512,input_shape=[100]))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(256))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(128))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(784))
generator.add(Reshape([28,28,1]))

generator.summary()


# **Discriminatory part**

# In[13]:


discriminator = Sequential()
discriminator.add(Dense(1,input_shape=[28,28,1]))
discriminator.add(Flatten())
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.5))
discriminator.add(Dense(128))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.5))
discriminator.add(Dense(64))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.5))
discriminator.add(Dense(1,activation='sigmoid'))


discriminator.summary()
# 1. Compiling the discriminator layer
# 2. Compiling the GAN 
# 
# NOTE : the generator layer is not compiled seperately because it gets trained as part of the combined model but training the discriminator is necessary because it is trained before the combined model.

# In[15]:


GAN =Sequential([generator,discriminator])
discriminator.compile(optimizer='adam',loss='binary_crossentropy')
discriminator.trainable = False

GAN.compile(optimizer='adam',loss='binary_crossentropy')

GAN.summary()

epochs = 5 #30
batch_size = 50
noise_shape=100
with tf.device('/gpu:0'):
 for epoch in range(epochs):
    print(f"Currently on Epoch {epoch+1}")
    
    
    for i in range(X_train.shape[0]//batch_size):
        
        if (i+1)%50 == 0:
            print(f"\tCurrently on batch number {i+1} of {X_train.shape[0]//batch_size}")
            
        noise=np.random.normal(size=[batch_size,noise_shape])
       
        gen_image = generator.predict_on_batch(noise)
        
        train_dataset = X_train[i*batch_size:(i+1)*batch_size]
       
        #training discriminator on real images
        train_label=np.ones(shape=(batch_size,1))
        discriminator.trainable = True
        d_loss_real=discriminator.train_on_batch(train_dataset,train_label)
        
        #training discriminator on fake images
        train_label=np.zeros(shape=(batch_size,1))
        d_loss_fake=discriminator.train_on_batch(gen_image,train_label)
        
        
        #training generator 
        noise=np.random.normal(size=[batch_size,noise_shape])
        train_label=np.ones(shape=(batch_size,1))
        discriminator.trainable = False
        
        d_g_loss_batch =GAN.train_on_batch(noise, train_label)
        
        
        
       
    #plotting generated images at the start and then after every 10 epoch
    if epoch % 10 == 0:
        samples = 10
        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, 100)))

        for k in range(samples):
            plt.subplot(2, 5, k+1)
            plt.imshow(x_fake[k].reshape(28, 28), cmap='gray')
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()

        
        
print('Training is complete')


# In[21]:


noise=np.random.normal(size=[10,noise_shape])

gen_image = generator.predict(noise)


# noise generated using np.random.normal is given to as input to the generator. In the next step the generator produces batches of meaningful alike image from the random distribution.

# In[22]:


plt.imshow(noise)
plt.title('How the noise looks')


# **Generator producing images from noise**

# In[23]:


fig,axe=plt.subplots(2,5)
fig.suptitle('Generated Images from Noise using GANs')
idx=0
for i in range(2):
    for j in range(5):
         axe[i,j].imshow(gen_image[idx].reshape(28,28),cmap='gray')
         idx+=1


Generator2 = Sequential()
Generator2.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
Generator2.add(BatchNormalization())
Generator2.add(ReLU())
Generator2.add(Reshape((7, 7, 256)))
Generator2.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
Generator2.add(BatchNormalization())
Generator2.add(ReLU())
Generator2.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
Generator2.add(BatchNormalization())
Generator2.add(ReLU())
Generator2.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False))


# Generate Model Summary:



Generator2.summary()


Discriminator2 = Sequential()
Discriminator2.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
Discriminator2.add(BatchNormalization())
Discriminator2.add(LeakyReLU())
Discriminator2.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
Discriminator2.add(BatchNormalization())
Discriminator2.add(Dense(1,activation='tanh')
                   
#Discriminator2.build(input_shape=(None,28,28,3))


Discriminator2.summary()


# Compiling the GAN:

# In[59]:


GAN2 =Sequential([Generator2,Discriminator2])
Discriminator2.compile(optimizer='adam',loss='binary_crossentropy')
Discriminator2.trainable = False


# In[60]:


GAN2.compile(optimizer='adam',loss='binary_crossentropy')


# GAN summary:

# In[61]:


GAN2.summary()


# Train your model - Mention epoch size, batch size, noise shape

# In[64]:


epochs = 5 #30
batch_size = 50
noise_shape=100


# In[65]:


with tf.device('/gpu:0'):
 for epoch in range(epochs):
    print(f"Currently on Epoch {epoch+1}")
    
    
    for i in range(X_train.shape[0]//batch_size):
        
        if (i+1)%50 == 0:
            print(f"\tCurrently on batch number {i+1} of {X_train.shape[0]//batch_size}")
            
        noise=np.random.normal(size=[batch_size,noise_shape])
       
        gen_image = generator.predict_on_batch(noise)
        
        train_dataset = X_train[i*batch_size:(i+1)*batch_size]
       
        #training discriminator on real images
        train_label=np.ones(shape=(batch_size,1))
        discriminator.trainable = True
        d_loss_real=discriminator.train_on_batch(train_dataset,train_label)
        
        #training discriminator on fake images
        train_label=np.zeros(shape=(batch_size,1))
        d_loss_fake=discriminator.train_on_batch(gen_image,train_label)
        
        
        #training generator 
        noise=np.random.normal(size=[batch_size,noise_shape])
        train_label=np.ones(shape=(batch_size,1))
        discriminator.trainable = False
        
        d_g_loss_batch =GAN.train_on_batch(noise, train_label)
        
        
        
       
    #plotting generated images at the start and then after every 10 epoch
    if epoch % 10 == 0:
        samples = 10
        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, 100)))

        for k in range(samples):
            plt.subplot(2, 5, k+1)
            plt.imshow(x_fake[k].reshape(28, 28), cmap='gray')
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()

        
        
print('Training is complete')


# Generate random noise and predict from generator:

# In[66]:


noise=np.random.normal(size=[10,noise_shape])

gene_image = Generator2.predict(noise)


# In[67]:


plt.imshow(noise)
plt.title('How the noise looks')


# Output in the form of Actual Images and Generated Images

# In[72]:


fig,axe=plt.subplots(2,5)
fig.suptitle('Generated Images from Noise using GANs')
idx=0
for i in range(2):
    for j in range(5):
         axe[i,j].imshow(gene_image[idx].reshape(28,28),cmap='gray')
         idx+=1

