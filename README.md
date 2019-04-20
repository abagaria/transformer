
# Language Modeling using Transformer Networks
### Adaptation of *Attention is All You Need* model for the language modeling task

Final test perplexity of my model: 262.9


1. Benefits of using transformer network as opposed to RNN network for language modeling:
    - Since transformers are built on top of simple matrix multiplications, they are much faster and parallizable than RNNs which have a sequential nature to their computations: the computation at timestep t depends on the computation at t-1 in an RNN. 
    - Transformers have empiracally been able to learn semantics in long sentences. Since RNN's (even fancier renditions like LSTMs and GRUs) suffer from vanishing gradients which make it difficult for them to learn long term dependencies in passages. 

2. Purposes of the query/key/value tuples in scaled dot-product attention:
    - The query, key, value terminology is borrowed from databases 
    - The idea is that the query and the key correspond to the input to a self-attention module whereas the value corresponds to the output or the label. In principle the query and the key can be different representations of our input but they tend to be the same. In the language modeling task, the inputs are the sentences and the outputs are the same sentences, shifted right by one. 
    - The scaled dot-product attention captures how much each word in the sequence must attend to every other word in the sequence.

3. Purpose of multi-headed attention:
Different projections of our input to different subspaces will allow us to capture different semantic relations between them. This is what the multi-headed attention allows us to do. With a single head, we would only be able to capture the semantic relatedness between words of a sentence in one such sense. 

4. Purpose of positional encoding: 
	- In RNNs, we were able to capture the fact that word word_{t+1} occured *after* word_t. This allowed us to capture the sequential/temporal nature of natural language. However, since transformers are simply parallel matrix multiplies, they don't have a natural way to encode such sequential dependencies. This is where positional encodings come in.
	- There is information embedded in the position that a word occurs in in natural language. For instance, the first word of a sentence is very often an article, while the last "word" is a period. 

	Reason sinusoid functions work for positional embeddings: 
	- Sinusoids ensure that words nearby each other have similar embedding values, while words far away have different values - which is the kind of information we want to capture with our positional embeddings.
	- Learned positional embeddings would probably work just as well (it was also reported as such in the Attention is all you need paper).

Hyperparameters used:
- number of heads: 3
- learning rate = 1e-4
- window_size = 20
- batch_size = 1
- embed_size = 120
- hidden_size = 120

Of the above hyperparameters, perhaps the most important was the learning rate. Going from 1e-3 to 1e-4 made a huge difference in my test perplexity. 

Test perplexity: 262.9

