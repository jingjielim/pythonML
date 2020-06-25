# Plot sigmoid funciton for visualization

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    
    ''' Sigmoid function 
    Input 
    z : float
        Any real number
        
    Returns
    output : float
        Real number between 0.0 and 1.0
    '''
    return 1.0 / (1 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='pink')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.yticks([0.0,0.5,1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.show()
