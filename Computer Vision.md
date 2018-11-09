# Computer-Vision
Deep Neural Networks

## Convolutional Neural Networks
Lets Talk about Convolutional Neural Networks
If you take an image 768 x 1024 x 1
turn it into a vector 786432 x 1
How many parameters would that be per hidden unit?
Yeah... Almost 800,000

Each output unit is a linear function of a localized subset of input units

768 x 1024 x 3
rows x cols x channels
rows x cols x channels x t

- Stride One Convolution:
  - Point wise multiplication and sum

- Stride Two Convolution

- Dilated Convolution

- [[-1 0 1],
  [-1 0 1].
  [-1 0 1]]
  Using this matrix, we have 3 vertical strips of black, grey, white.
  You pass this all over the image.
  I think it will result in an image that shows edges of things only and the rest becomes grey.
  
  Now, if you take an image 
  32 x 32 x 3
  *
  3 x 3 x 3 x 3
  and then... Not sure...
  
### Convolution as Matrix Multiplication
  
- 1D convolution, stride one, padding zero
- 1D convolution, stride one, padding one
- 1D convolution, stride two, padding one

### Feature Pooling Layers
- Average Pooling
  - 2-by-2 average pool with stride 2
  - 2-by-2 max pool with stride 2
  
### Activation Functions
- sigmoid
- hyperbolic tangent
- Rectified Linear Unit (ReLU)

Must Watch a Video here to better understand
https://www.youtube.com/watch?v=FmpDIaiMIeA

CNN's 
if you feed them a bunch of images of faces. it will learn.
Because it is multi learned,
 - The first layer will learn things like basic things edges, dots, bright spots, dartk 
 - The second layer will learn things like things that are recognizable like eyes, noses, mouths
 - The third layer will be things that look like faces.
A similar thing will happen with images of cars.
CNN's can even learn to play video games by forming patterns of pixels as they appear on screen and learning what is the best action to take.

Lets look at a toy ConvNet: X's and O's
Trickier cases
translation, scaling, rotation, weight

ConvNets match pieces of the image rather than the whole thing.
By breaking it into parts or features, then it becomes much more clear if two things are similar

Example of these features are little mini images
[pic at 4:40]

The math behind matching the features to the image is called filtering.
The way this is done is that the feature is lined up agaisnt the image and then one by one the pixels are compared.
1. Liine up the feature and the image patch
2. Multiply each image pixel by the corresponding feature pixel.
3. Add them up.
4. Divide by the total number of pixels in the feature.

Convolution is the repeated application of that feature.
Then we repeat the process with other features.

This act of convolving an image with a bunch of features,  creates a stack of filtered images that we'll call a convolution layer.
Its an operation that we can stack with others.
In convolution, one image becomes a stack of filtered images.

Pooling: Shrinking the image stack
This is how we shrink the image down.
This makes sense because shrinking a 9000x9000 image is too big.
Pooling doesnt care where in the window the value occurs, it makes it less sensitive to the position. That means if youre looking for something that might be a little to the left or a little to the right, it will still get picked up.

So if we do max pooling with all of our stack of filtered images.
we get smaller filtered images.

Normalization
Keep the math from breaking by tweaking each of the values just a bit.
Everywhere where there is a negative value, change it to 0.
THATS what the Rectified Linear Units (ReLUs) Do. 

Layers get stacked.
Stack up the Convolution layer, the ReLU layer and the pooling layer.
We can repeat these stacks. And make a sandwich.
Each time the image gets more filtered as it goes through convolution layers.
It gets smaller as it goes through pooling layers.

Now the final layer in our toolbox, is called the fully connected layer.
Here, every value gets a vote as to what the answer is going to be.
[14:50]

So in a fully connected layer, a list of feature values becomes a list of votes.

A list of votes looks a whole lot like a list of feature values.
you can have intermediate categories that are not your final vote. These are called hidden units.
In the end they will vote for an X or an O.

Learning
Q. Where do all the magic numbers come from?
   - Features in convolutional layers
   - Voting weights in fully connected layers
A. Backpropagation

They are all learned. The Deep neural network does these on its own.
The error in the final answer is used to determine how much the network adjusts and changes.

The error signal helps drive a process called gradient descent.
The amount that they are adjusted depends on the amount of error.
You want to find the direction where the slope is going downhill.

Doing that many times, helps all the values accross all ofthe weights help settle in, into whats called a minimum.

Hyperparameters (knobs)
Things the designer has to decide
- Convolution
  - Number of features
  - Size of features
- Pooling
  - Window size
  - Window stride
- Fully Connected
  - Number of neurons

There are some common practices that tend to work better than others.
Alot of the advances in CNN is in getting combination of these that work really well.

Architecture
There are other decisions that the designer needs to make 
- Hopw many of each type of layer?
- In what order?
- Can we design new types of layers entirely.

Not just images.
Any 2D or 3D data.
Things closer togetherare more closely related than things far away.

If you take something like Sound.
for each piece of time the timestep before and after are more closely related than those far away.
Chop it up ionto diff freq bands. Those freq bands are the ones closer together. The order matters.

The sound then looks like an image and you can use CNN on them.

You can do something similar with Text, where the position in the sentence becoes the column and the rows are words in a dictionary.
In this case, its hard to argue that order matters.. so the trick hgere is to take a window that spans the entire coluimn and then slide it left to right. it captures all the words but only a few positions in the sentense.\

The limitation of CNN's. they are designed to capture local "spatial" patterns in data.
If the data cant be made to look like an image, ConvNets are less useful.

An example of this not6 working.. could be customer data:
Name, age, address, email, purchases, browsing activity,..

As a rule of thumb, If you data is just as useful after swapping any of your columns with each other, then you can't use convolutional Neural Networks.

In a nutshell:
Convolutional NN are greatat finding patterns and using them to classify images.
If you can make your problem looks like finding cats on the internet, then they are a huge asset.

IF tyou were to use linear activation functions then... the output is outpuiting a linear function of the inpout.
If you dont have an activation function, then no matter how many layers in your neural network, all its doing is computing a linear function. and you might as well have no hidden layers.
This model is the no more expressive than standard logistic regression without any hidden layer.
The take home is that a linear hidden layer is more or less useless because the composition of 2 linear functions is itself a linear function. So unless you throw non-linearity in there, you are not computing more interesting functions even as you go deeper in the network. 
There is only one place you might use a linear function. If you are using ML on a regression problem. If you are trying to predict housing prices.


Torch

Getting an idea behind the softmax function:
```
import numpy as np

def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps)
    
x = softmax([1, 2, 3])
y = softmax([1000, 2000, 3000])

print('x = ', x)
print('y = ', y)
```
Torch example. A Multilayer perceptron
```
import torch
from torch import Tensor
#Underscores denote in-place operations
x = Tensor(10).normal_()
W1 = Tensor(20,10).normal_()
W2 = Tensor(5,20).normal_()
b1 = Tensor(20).normal_()
b2 = Tensor(5).normal_()

h = torch.sigmoid(W1.matmul(x) + b1)
y_pred = torch.sigmoid(W2.matmul(h) + b2)
```

Using GPU cuda cores:
Instructions:
1. Download Anaconda
2. Download Nvidea Cuda Core Package
```
*** add cudaPython
```


Tensorflow
