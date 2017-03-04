'''
Created on Feb 8, 2017

@author: Mohsin Qureshi
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
+ 0.5 * np.random.randn(100 , 1) #returns a sample from the normal distribution of mean 0 and sd 1
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

def distMat(a, b):
    fixb = np.transpose(b[np.newaxis,:])
    dists = ((np.square(a[:,np.newaxis]-fixb).sum(axis=2)))
    return dists

def resMat(dists, k):
    tfdists = tf.constant(dists)
    topdists = tf.nn.top_k(tf.transpose(-tfdists), k, sorted=True)
    sess = tf.Session()
    print("This is the distance matrix")
    print(sess.run(tf.transpose(tfdists)))
    print("These are the indexes of the closest %d x's" %k)
    ind = sess.run(topdists.indices)
    ind = np.sort(ind)
    print(ind)
    ind2 = tf.constant(ind)
    print("This is the shape of the output")
    print(dists.shape)
    flipSize = np.flipud(dists.shape)
    print(np.flipud(dists.shape))
    print("I need to pad the indexes so that sparse_to_dense can take the input")
    ind3 = ind.flatten()
    inputsize = ind3.shape[0]
    print(ind3.shape[0])
    apparr = []
    for i in range(0, inputsize//k):
        for j in range(0, k):
            apparr.append(i)
    print(apparr)  
    w = np.vstack((apparr,ind3))
    print(w)
    w1 = np.transpose(w)
    print(np.transpose(w)) 
    print(flipSize)
    res = tf.sparse_to_dense(w1, flipSize, 1/k, 0)
    print("This is the responsibility matrix")
    print(sess.run(res))
    return sess.run(res)
 
def predOut(Y, resMat):
    sess = tf.Session()
    print(Y.shape)
    print(resMat.shape)
    y_predicted = tf.matmul(tf.transpose(Y),tf.cast(tf.transpose(resMat), tf.float64))
    sess = tf.Session()
    result = sess.run(y_predicted)
    print("This is the predicted y output")
    print(result)
    return result

Xs = np.linspace(0.0, 11.0, num =1000)
Ys = predOut(trainTarget, resMat(distMat(trainData, Xs),5))
plt.plot(trainData, trainTarget, 'bo')
plt.plot(np.squeeze(Xs[np.newaxis,:]), np.squeeze(Ys), 'r')


#validDataYs = predOut(validData[np.newaxis,:], resMat(trainData, Xs), 1)

sess = tf.Session()
#print("reducing the means between:")
print(np.transpose(Ys))


#print("This is the trainData")
#print(trainData)
print("This is the validData")
print(validData)

valids = predOut(trainTarget, resMat(distMat(trainData, validData),5))

print("This is predicted output")
print(np.squeeze(valids))
print("This is the actual Y output")
print(np.squeeze(np.transpose(validTarget)))
#now i want to subtract valids from validTargets.



MSE = tf.nn.l2_loss(np.squeeze(valids)-np.squeeze(np.transpose(validTarget)))
MSE2 = tf.reduce_sum(tf.pow(np.squeeze(valids)-np.squeeze(np.transpose(validTarget)), 2))/(10)
print("The MSE is")
print(sess.run(MSE))
print("The MSE2 is")
print(sess.run(MSE2))

'''
meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(np.transpose(Ys) - trainTarget), 
                                            reduction_indices=1, 
                                            name='squared_error'), 
                              name='mean_squared_error')

#loss = tf.div(tf.reduce_sum(diff_sqr,0), y.get_shape().as_list()[0] * 2)

print(sess.run(meanSquaredError))
'''
plt.show()