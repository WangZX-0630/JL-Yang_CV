from sklearn.metrics import *
import numpy as np
pic1=np.random.randint(0,3,(512,512))
pic2=np.random.randint(0,3,(512,512))
kappa=cohen_kappa_score(pic1.flatten(),pic2.flatten())
print("kappa是{}".format(kappa))