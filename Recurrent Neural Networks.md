# Recurrent Neural Networks

Stanford University School of Engineering Lecture 10

Batch normalization is very important.
RezNet have these shortcut connections.

The gradient flow in the backward pass... then gradient will split and fork. 
These connections give a sort of gradient superhighway for the gradients to flow back through the network.


-> DenseNet is a more recent architecture.
-> FractalNet is also new.

Managing gradient flow in your architectures is something we have seen more in the past couple of years.

AlexNet ~62M parameters
59 of the 62M parameters are in the fully connected layers.

Recap Done.

RNN - So far, we've seen vanilla feed forward networks.
one to one.
We recieve that input. a fixed sized object.
produces a single output.

We want more flexibility.
with rnn we have more opportunities.
we can do one to many modles.

One to Many --> The input might be an image --> and the output can be a caption like a sequence of words.
Many to one --> A piece of text --> We want to say what the sentiment is. + or  -
            --> A video that has multiple frames --> make a classification decision as to the activity or action
            
Many to Many --> Machine Translation. Sentence in english --> Sentence in french
Many to Many --> Video sequence --> A decision for each element fo the vid seq. a classification decision along every frame of vid.

RNN allow all these.

Read an input --> update its hidden state --> produce an output.

We can process a sequence of vectors x by applying a recurrence forumula at every time step:

ht = fw(ht-1, xt)

ht = new state
fw = some function with parameters W
ht-1 = old state
xt = input vector at some time step

ht will then be passed into the same function as we need the next input xt+1
If we wanted to produce some output at each time step, we can attach some additional fully connected layers that feed in this ht and make that deicision based on the hidden stage at every timestep.

we use the same weights at every timestep?

Vanilla RNN?
ht = fw(ht-1, xt)
ht=tanh(Whhht-1+wxhxt)
yt=Whyht



















`
