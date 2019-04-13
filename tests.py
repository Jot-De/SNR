
# coding: utf-8

# In[ ]:


from keras.applications.vgg16 import VGG16
model = VGG16()
print(model.summary())


# In[ ]:


# Check the trainable status of the individual layers
vgg_conv = VGG16(weights='imagenet', include_top=True)
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


# In[ ]:


from keras.utils.vis_utils import plot_model
model = VGG16()
plot_model(model, to_file='vgg.png')

