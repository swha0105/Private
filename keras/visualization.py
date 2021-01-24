#%%
from keras.preprocessing import image
import numpy as np

img = image.load_img('/home/ha/data/keras/small/test/cats/cat.1700.jpg', target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

img_tensor.shape

#%%
import matplotlib.pyplot as plt 

plt.imshow(img_tensor[0])
plt.show()

#%%

from keras import models

layer_outputs = [layer.get_output_at( ) for layer in model.layers[:8]]
#layer_output = layer.get_output_at(-1)
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

#%% 
## 5.28
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
first_layer_activation.shape

#%%
## 5.29
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,4], cmap='viridis')

#%%
