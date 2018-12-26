import numpy as np
from layers import softmax
import model
#test one image
def test(model, x):
    result = model.prediction(x[None, :, :, :])
    scores = softmax(result)
    top_three = np.argsort(scores)[:, -3:]
    top_three = top_three.reshape(3)
    for i in range(3):
        print('ID:%d Confidence:%f'%(top_three[-1 - i], scores[top_three[-1 - i]]))
    
    