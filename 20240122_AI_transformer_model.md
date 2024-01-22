# Attention model

Attention models (proposed in 2014) are designed to cope with longer sentences than Seq2Seq architectures cope with well, eg > 30 or 40 words. It works by having "attention gates" with weights that quantify how important the local context for each word is, ie how it depends on preceding and subsequent words. It captures that the closer neighbours are more important than distant ones - so effectively allowing a long sentence to be broken down into smaller parts. This local importance is stored in a context vector.

In language translation models, the basic idea is that the output of the cell ‘points’ to the previously encountered word with the highest attention score. However, the model also uses the standard softmax classifier over a vocabulary V so that it can predict output words that are not present in the input in addition to reproducing words from the recent context. The attention architecture is to have a sequential decoder (Seq2Seq), but in addition to the previous cell’s output and hidden state, you also feed in a context vector c. Because we are using it with training data (where we know the desired output) the desired output sentence as well as the input sentence is used to train the context vector - using a "softmax" activation function. https://blog.paperspace.com/seq-to-seq-attention-mechanism-keras/ 

Attention models can be used for image captioning which is describing an image using text (ie they split the image up into parts and identify what is going in them - that is relevant to the supervised learning problem, eg where cancer is on a X ray image), speech recognition, self-driving cars, machine translation, document summarisation. https://distill.pub/2016/augmented-rnns/

Attention is used in the Transformer in three places:

    - Self-attention in the Encoder — the input sequence pays attention to itself
    - Self-attention in the Decoder — the target sequence pays attention to itself (eg input = a sentence in french and output = english sentence for a french to english translation)
    - Encoder-Decoder-attention in the Decoder — the target sequence pays attention to the input sequence.

So the attention model:

    - operates with tokens as words and sequences as a set (sentence) of words; it passes through the sentences one word at a time (ie it operates at word level, not character level)
    - runs through a sequence (ie set of words in French) paying attentions to each word's context (using "attention") - this is the encoder
    - it also runs through the translated sequence of words paying attention to each words context - this is the decoder
    - it then is able to train a neural network to map between the encoder and decoder - the Encoder-Decoder-attention model.
    - then when we ask it to "predict": it encounters a new French sentence to translate it applies these 3 sub-models to generate an English translated sentence.



```python
# Now import the libraries and algorithms required:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm  # This displays a progress bar.
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay  # To plot the ROC curve.
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Activation, Dropout, Embedding, BatchNormalization
# from keras.utils import # utils contains a set of functions: set_random_seed(), split_dataset, get_file, Progbar, Sequence, to_categorical, to_ordinal, normalize
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from keras import Input, Model  # extra commands required for the Seq2Seq model in PART 6.
from keras.models import load_model  # extra command required for the Seq2Seq model in PART 6.

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
```

### Preprocess the Dataset

The dataset has Unicode characters, which have to be normalized.

Moreover, all the tokens (ie words) in the sequences have to be cleaned using the regular expressions (re) library.

Remove unwanted spaces, include a space between every word and the punctuation following it (to differentiate between both), replace unwanted characters with spaces, and append 'start' and 'end' tokens to specify the start and end of a sequence.

Encapsulate the unicode conversion (ie text) in a function unicode_to_ascii() and sequence preprocessing in a function preprocess_sentence().


```python
import unicodedata
import re  # regular expression. Powerful basic text processing functions.

# Convert the unicode sequence to ascii
def unicode_to_ascii(s):

  # Normalize the unicode string and remove the non-spacing mark. The python string join() method takes all items in an iterable and joins them into one string. A string must be specified as the separator. Here '' (ie nothing) is the separator.
  return ''.join(c for c in unicodedata.normalize('NFD', s) # Return the normal form NFD for the Unicode string s. Normal form D (NFD) is also known as canonical decomposition, and translates each character into its decomposed form. It will remove accents.
      if unicodedata.category(c) != 'Mn')  # Returns the general category assigned to the character c as string.

# Preprocess the sequence. Define a function.
def preprocess_sentence(w):

  # Clean the sequence
  w = unicode_to_ascii(w.lower().strip())

  # Create a space between word and the punctuation following it. Use functionality from the re (regular expression) library.
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  # Add a start and stop token to detect the start and end of the sequence
  w = '<start> ' + w + ' <end>'
  return w
```


```python
# Prepare the Dataset
# Next, prepare a dataset out of the raw data we have. Create word pairs combining the English sequences and their related French sequences.

path = 'Documents/Mike/MOOCs/NLP_training_data_2023/fra.txt'  # "fra.txt" 
num_examples = 30000   # Consider 50k examples (ie sentences). Reduce the number of data samples required to train the model. Employing the whole dataset will consume a lot more time for training the model.

# Create the Dataset
def create_dataset(path, num_examples):

  lines = open(path, encoding='UTF-8').read().strip().split('\n')

# Loop through lines (sequences) and extract the English and French sequences. Store them as a word-pair
  word_pairs = [[preprocess_sentence(w) for w in l.split('\t', 2)[:-1]]  for l in lines[:num_examples]]
  return zip(*word_pairs)  # The zip() function takes iterables (can be zero or more), aggregates them in a tuple, and returns it. The asterisk (*) prefix in the variable object is used to tell python that it's a packing argument, ie you pass multiple arguments into the function.
```


```python
en, fra = create_dataset(path, num_examples)  # Check if the dataset has been created properly.
print(en[-1])
print(fra[-1])
```

    <start> tom must be tired . <end>
    <start> tom doit etre fatigue . <end>
    

Now tokenize the sequences. Tokenization is the mechanism of creating an internal vocabulary comprising English and French tokens (i.e. words), converting the tokens (or, in general, sequences) to integers, and padding them all to make the sequences possess the same length. All in all, tokenization facilitates the model training process.

- note that here tokenization is at word level (not character level).


```python
# Create a function tokenize() to encapsulate all the above-mentioned requirements.
# Keras tokenizer class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each 
# token could be binary, based on word count, based on tf-idf...
# By default, all punctuation is removed, turning the texts into space-separated sequences of words (words may include the ' character). These sequences are then split into lists of tokens. They will then be indexed or vectorized.

import tensorflow as tf

# Convert sequences to tokenizers
def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  
  # Convert sequences into internal vocab
  lang_tokenizer.fit_on_texts(lang)

  # Convert internal vocab to numbers
  tensor = lang_tokenizer.texts_to_sequences(lang)

  # Pad the tensors to assign equal length to all the sequences. This is zero padding so all words are the same length.
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

  return tensor, lang_tokenizer
```


```python
# Create a function tokenize() to encapsulate all the above-mentioned requirements.
# Keras tokenizer class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each 
# token could be binary, based on word count, based on tf-idf...
# By default, all punctuation is removed, turning the texts into space-separated sequences of words (words may include the ' character). These sequences are then split into lists of tokens. They will then be indexed or vectorized.

import tensorflow as tf

# Convert sequences to tokenizers
def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  
  # Convert sequences into internal vocab
  lang_tokenizer.fit_on_texts(lang)

  # Convert internal vocab to numbers
  tensor = lang_tokenizer.texts_to_sequences(lang)

  # Pad the tensors to assign equal length to all the sequences. This is zero padding so all words are the same length.
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

  return tensor, lang_tokenizer
```


```python
# Load the tokenized dataset by calling the create_dataset() and tokenize() functions.

# Load the dataset
def load_dataset(path, num_examples=None):
 
  # Create dataset (targ_lan = English, inp_lang = French)
  targ_lang, inp_lang = create_dataset(path, num_examples)

  # Tokenize the sequences
  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
```


```python
# Consider 50k examples. Actually I'm cutting down num_examples to 30k from 50k - quite a big reduction (for speed).
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
# The max_length of both the input and target tensors is essential to determine every sequence's maximum padded length.
# Here the target language is English, the input is French.
```


```python
print(max_length_targ,'    ',max_length_inp)
```

    9      17
    


```python
# Create the Dataset

# Segregate the train and validation datasets.
# Create training and validation sets using an 80/20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
```

    24000 24000 6000 6000
    


```python
# Validate the mapping that’s been created between the tokens of the sequences and the indices.

# Show the mapping b/w word index and language tokenizer
def convert(lang, tensor):
  for t in tensor:
    if t != 0:
      print ("%d ----> %s" % (t, lang.index_word[t]))
      
print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print ()
print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])
```

    Input Language; index to word mapping
    1 ----> <start>
    16 ----> le
    461 ----> stylo
    5 ----> est
    291 ----> casse
    3 ----> .
    2 ----> <end>
    
    Target Language; index to word mapping
    1 ----> <start>
    17 ----> the
    442 ----> pen
    10 ----> is
    668 ----> broken
    3 ----> .
    2 ----> <end>
    

Initialize the Model Parameters

With the dataset in hand, start initializing the model parameters.

BUFFER_SIZE: Total number of input/target samples. In our model, it’s 24,000.
BATCH_SIZE: Length of the training batch. ie 64 sentences to translate at a time. Working in batches is more efficient than 1 sentence at a time.
steps_per_epoch: The number of steps per epoch. Computed by dividing BUFFER_SIZE by BATCH_SIZE.
embedding_dim: Number of nodes in the embedding layer. I think this is the amount of dimensions used to compress each word in a sentence to.
units: Hidden units in the network.
vocab_inp_size: Length of the input (French) vocabulary.
vocab_tar_size: Length of the output (English) vocabulary.


```python
# Essential model parameters
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64 # Feed the model batches of 64 sentences at a time. Used to make the training process more efficient/fast.
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256  # The number of dimensions that each token (word) gets compressed to as a numeric vector by the embedding algorithm.
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1  # 7472 (french)
vocab_tar_size = len(targ_lang.word_index) + 1    # 4299 (english)
```


```python
# Next, call the tf.data.Dataset API and create a proper dataset.

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
```


```python
# Validate the shapes of the input and target batches of the newly-created dataset.

# Size of input and target batches
example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape   # (TensorShape([64, 17]), TensorShape([64, 9])) 17 and 9 denote the maximum padded lengths of the input (French) and target (English) sequences
```




    (TensorShape([64, 17]), TensorShape([64, 9]))



Encoder Class

The first step in creating an encoder-decoder sequence-to-sequence model (with an attention mechanism) is creating an encoder. For the application at hand, create an encoder with an embedding layer followed by a GRU (Gated Recurrent Unit) layer. The input goes through the embedding layer first and then into the GRU layer. The GRU layer outputs both the encoder network output and the hidden state.

Enclose the model’s init() and call() methods in a class Encoder.

In the method, init(), initializes the batch size and encoding units. Add an embedding layer that accepts vocab_size as the input dimension and embedding_dim as the output dimension. Also, add a GRU layer that accepts units (dimensionality of the output space) and the first hidden dimension.

In the method call(), define the forward propagation that has to happen through the encoder network.

Moreover, define a method initialize_hidden_state() to initialize the hidden state with the dimensions batch_size and units.


```python
# Encoder class
class Encoder(tf.keras.Model):  # our new Encoder class inherits attributes and methods from the tf.keras.Model object.
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):  # The attributes that we are adding in this class.
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units

    # Embed the vocab to a dense embedding 
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) # Add an embedding layer that accepts vocab_size as the input dimension and embedding_dim as the output dimension.

    # GRU Layer
    # glorot_uniform: Initializer for the recurrent_kernel weights matrix, 
    # used for the linear transformation of the recurrent state
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform') # Add a GRU layer that accepts units (dimensionality of the output space) and the first hidden dimension.

  # Encoder network comprises an Embedding layer followed by a GRU layer
  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  # To initialize the hidden state as zero's.
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
```


```python
# Call the encoder class to check the shapes of the encoder output and hidden state.

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE) # Build the Encoder model object, passing it our parameters (eg vocab_inp_size =  7472 french sentences).

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

# Output
# Encoder output shape: (batch size, sequence length, units) (64, 17, 1024)
# Encoder Hidden state shape: (batch size, units) (64, 1024)
```

    Encoder output shape: (batch size, sequence length, units) (64, 17, 1024)
    Encoder Hidden state shape: (batch size, units) (64, 1024)
    

Attention Mechanism Class

This step captures the attention mechanism.

Compute the sum (or product) of the encoder’s outputs and decoder states.
Pass the generated output through a fully-connected network.
Apply softmax activation to the output. This gives the attention weights.
Create the context vector by computing the weighted sum of attention weights and encoder’s outputs.

Everything thus far needs to be captured in a class BahdanauAttention. Bahdanau Attention is also called the “Additive Attention”, a Soft Attention technique. As this is additive attention, we do the sum of the encoder’s outputs and decoder hidden state (as mentioned in the first step).

This class has to have init() and call() methods.

In the init() method, initialize three Dense layers: one for the decoder state ('units' is the size), another for the encoder’s outputs ('units' is the size), and the other for the fully-connected network (one node).

In the call() method, initialize the decoder state (s0) by taking the final encoder hidden state. Pass the generated decoder hidden state through one dense layer. Also, plug the encoder’s outputs through the other dense layer. Add both the outputs, encase them in a tanh activation and plug them into the fully-connected layer. This fully-connected layer has one node; thus, the final output has the dimensions batch_size * max_length of the sequence * 1.

Later, apply softmax on the output of the fully-connected network to generate the attention weights.

Compute the context_vector by performing a weighted sum of the attention weights and the encoder’s outputs.


```python
# Attention Mechanism
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):  # Our new class's attributes. They can be different from the parent class - here the "units" attribute has been added.
    super(BahdanauAttention, self).__init__()  # super() tells the BahdanauAttention class that it inherits no attribute values from the parent class. The tensorflow BahdanauAttention Implements Bahdanau-style (additive) attention.
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # values shape == (batch_size, max_len, hidden size)

    # we are doing this to broadcast addition along the time axis to calculate the score
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
```


```python
    # Validate the shapes of the attention weights and its output.
attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

# Output
# Attention result shape: (batch size, units) (64, 1024)
# Attention weights shape: (batch_size, sequence_length, 1) (64, 17, 1)
# sample_hidden here is the hidden state of the encoder, and sample_output denotes the encoder’s outputs
```

    Attention result shape: (batch size, units) (64, 1024)
    Attention weights shape: (batch_size, sequence_length, 1) (64, 17, 1)
    

Decoder Class

This step encapsulates the decoding mechanism (ie translation from french to english). The Decoder class has to have two methods: init() and call().

In the init() method, initialize the batch size, decoder units, embedding dimension, GRU layer, and a Dense layer. Also, create an instance of the BahdanauAttention class.

In the call() method:

Call the attention forward propagation and capture the context vector and attention weights.
Send the target token through an embedding layer.
Concatenate the embedded output and context vector.
Plug the output into the GRU layer and then into a fully-connected layer.


```python
# Decoder class
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Used for attention
    self.attention = BahdanauAttention(self.dec_units) # Use the class defined earlier.

  def call(self, x, hidden, enc_output):
    # x shape == (batch_size, 1)
    # hidden shape == (batch_size, max_length)
    # enc_output shape == (batch_size, max_length, hidden_size)

    # context_vector shape == (batch_size, hidden_size)
    # attention_weights shape == (batch_size, max_length, 1)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights
```


```python
# Validate the decoder output shape.

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

# Output : Decoder output shape: (batch_size, vocab size) (64, 4299)
```

    Decoder output shape: (batch_size, vocab size) (64, 4299)
    


```python
# Define the optimizer and loss functions.
# As the input sequences are being padded with zeros, nullify the loss when there’s a zero in the real value.

# Initialize optimizer and loss functions
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# Loss function
def loss_function(real, pred):

  # Take care of the padding. Not all sequences are of equal length.
  # If there's a '0' in the sequence, the loss is being nullified
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
```


```python
# Train the Model

# Checkpoint your model’s weights during training. This helps in the automatic retrieval of the weights while evaluating the model.

import os

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
```

Next, define th training procedure. First, call the encoder class and procure the encoder outputs and final hidden state. Initialize the decoder input to have the token spread across all the input sequences (indicated using the BATCH_SIZE). Use the teacher forcing technique to iterate over all decoder states by feeding the target as the next input. This loop continues until every token in the target sequence (English) is visited.

Call the decoder class with decoder input, decoder hidden state, and encoder’s outputs. Procure the decoder output and hidden state. Compute the loss by comparing the real against the predicted value of the target. Fetch the target token and feed it to the next decoder state (concerning the successive target token). Also, make a note that the target decoder hidden state will be the next decoder hidden state.

After the teacher forcing technique gets finished, compute the batch loss, and run the optimizer to update the model's variables.


```python
@tf.function  # This will help you create performant and portable models, eg use tf.function in higher-level functions, like a training loop.
def train_step(inp, targ, enc_hidden):
  loss = 0

  # tf.GradientTape() -- record operations for automatic differentiation
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    # dec_hidden is used by attention, hence is the same enc_hidden
    dec_hidden = enc_hidden

    # <start> token is the initial decoder input
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):

      # Pass enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      # Compute the loss
      loss += loss_function(targ[:, t], predictions)

      # Use teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  # As this function is called per batch, compute the batch_loss
  batch_loss = (loss / int(targ.shape[1]))

  # Get the model's variables
  variables = encoder.trainable_variables + decoder.trainable_variables

  # Compute the gradients
  gradients = tape.gradient(loss, variables)

  # Update the variables of the model/network
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss
```

Now initialize the actual training loop. Run your loop over a specified number of epochs. First, initialize the encoder hidden state using the method initialize_hidden_state(). Loop through the dataset one batch at a time (per epoch). Call the train_step() method per batch and compute the loss. Continue until all the epochs have been covered.

MIKE - the next step takes hours to run as each of the 30 epochs takes over 8 minutes. To make it run faster reduce epochs down from 30.


```python
import time

EPOCHS = 30

# Training loop
for epoch in range(EPOCHS):
  start = time.time()

  # Initialize the hidden state
  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  # Loop through the dataset
  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

    # Call the train method
    batch_loss = train_step(inp, targ, enc_hidden)

    # Compute the loss (per batch)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # Save (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  # Output the loss observed until that epoch
  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
```

### Test the Model

Now define your model evaluation procedure. First, take the sentence given by the user into consideration. This has to be given in the French language. The model now has to convert the sentence from French to English.

Initialize an empty attention plot to be plotted later on with max_length_target on the Y-axis, and max_length_input on the X-axis.

Preprocess the sentence and convert it into tensors.

Then plug the sentence into the model.

Initialize an empty hidden state which is to be used while initializing an encoder. Usually, the initialize_hidden_state() method in the encoder class gives the hidden state having the dimensions batch_size * hidden_units. Now, as the batch size is 1, the initial hidden state has to be manually initialized.

Call the encoder class and procure the encoder outputs and final hidden state.

By looping (one word/token at a time) over max_length_targ, call the decoder class wherein the dec_input is the token, dec_hidden state is the encoder hidden state, and enc_out is the encoder’s outputs. Procure the decoder output, hidden state, and attention weights.

Create a plot using the attention weights. Fetch the predicted token with the maximum attention. Append the token to the result and continue until the token is reached.

The next decoder input will be the previously predicted index (concerning the token).

Add the following code as part of the evaluate() function.



```python
import numpy as np

# Evaluate function -- similar to the training loop
def evaluate(sentence):  # The evaluate function is called with the sentence to evaluate.

  # Attention plot (to be plotted later on) -- initialized with max_lengths of both target and input
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  # Preprocess the sentence given
  sentence = preprocess_sentence(sentence)

  # Fetch the indices concerning the words in the sentence and pad the sequence
  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  # Convert the inputs to tensors
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  # Loop until the max_length is reached for the target lang (ENGLISH)
  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # Store the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    # Get the prediction with the maximum attention
    predicted_id = tf.argmax(predictions[0]).numpy()

    # Append the token to the result
    result += targ_lang.index_word[predicted_id] + ' '

    # If <end> token is reached, return the result, input, and attention plot
    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # The predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot
```

### Plot and Predict

Define the plot_attention() function to plot the attention statistics.


```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()
```

Define a function translate() which internally calls the evaluate() function.


```python
# Translate function (which internally calls the evaluate function)
def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  plot_attention(attention_plot, sentence.split(' '), result.split(' '))
```

Restore the saved checkpoint to the model.


```python
# Restore the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
```


```python
translate(u"As tu lu ce livre?")
```

The actual translation is "Have you read this book?" - so the translation is mostly correct but is missing the word "book".

Now try translating another sentence.


```python
translate(u"Comment as-tu été?")
```

The actual translation is "How have you been?"

MIKE: As can be inferred, the predicted translations are in proximity to the actual translations. When training the model, I cut down num_examples (ie phrases) to 30k from 50k - quite a big reduction (for speed). The recommendation is to use 50k.

Conclusion

This demonstrates what Attention Mechanism is all about. It fares better than a general encoder-decoder sequence-to-sequence model. It shows how to train a neural machine translation model to convert sentences from French to English


```python

```
