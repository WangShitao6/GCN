from tensorflow.keras.layers import Input,add
from tensorflow.keras.models import Model
import numpy as np
inputa = Input(shape=(10,10),name='inputa')
inputb = Input(shape=(10,10),name='inputb')
output = add([inputa,inputb])
model = Model(inputs=[inputa,inputb],outputs=[output])

a = np.array(range(200)).reshape((-1,10,10))
print('shape {}:{}'.format(a.shape,a))

b = np.array(range(200)).reshape((-1,10,10))
print('shape {}:{}'.format(b.shape,b))
c=model.predict([a,b])
print(c)
