
# BERT

Final test perplexity of my model: 535.27


1. Benefits of using BERT over simple embedding layer:
	BERT can create much richer word representations than simple word embeddings
	This has to do with the fact that BERT leverages many layers to learn word embeddings and more importantly pays attention to the context surrounding each word.

2. Purposes of masking as described in the paper:
	- By masking a random selection of words, we are training pur language model to predict a given word not based only on words that precede it (as in the previous assignment), but on all the context that surrounds it. This allows us to leverage more information than a left-right language model and hence learn richer word embeddings.
	- We only mask 80% of the words selected because in test time (or fine-tuning), the model is not going to see <MASK> tokens - to mitigate this mismatch, they replace some of the MASK tokens with random words or leave them unchanged. 

3. To solve the MCQ problem, we can treat the input passage and all the options as the context vector. Then we can ask the model to predict the next sequence from a vocab that includes only the possible indices of the possible answers.

Hyperparameters used:
- number of heads: 3
- number of transformer boxes: 2
- learning rate = 1e-4
- window_size = 100
- batch_size = 1
- embed_size = 150
- hidden_size = 150

Of the above hyperparameters, perhaps the most important was the learning rate. Going from 1e-3 to 1e-4 made a huge difference in my test perplexity. 

Test perplexity: 535.27

I generated embedding plots for the words in the specified in the problem statement. I noticed that words that were used in similar context tended to cluster together, while words with vastly different usages were far apart.

