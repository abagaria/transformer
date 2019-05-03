import sys
import torch
import pickle
import numpy as np
import pdb

from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from utils import get_sentences
from model import LanguageModel
from hyperparameters import Hyperparameters
from dataset import LMDataset


def plot_embeddings(texts, embeddings, show_plot=False, plot_name='plot.png'):
    """
    Uses MDS to plot embeddings (and its respective sentence) in 2D space.

    Inputs:
    - texts: A list of strings, representing the sentences
    - embeddings: A 2D numpy array, [num_sentences x embedding_size],
        representing the relevant word's embedding for each sentence
    """
    mds = MDS(n_components=2)
    embeddings = mds.fit_transform(embeddings)

    plt.figure(1)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], color='navy')
    for i, text in enumerate(texts):
        plt.annotate(text, (embeddings[i, 0], embeddings[i, 1]))
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    plt.savefig(plot_name, dpi=100)
    if show_plot:
        plt.show()


def create_embeddings(word_set):
    chosen_windows = []
    word_indices = []

    for window in dataset.training_data:
        for chosen_word in word_set:
            if chosen_word in window:
                word_indices.append(window.index(chosen_word))
                chosen_windows.append(window)

    # TODO: Load the model. Give these sentences as input, and obtain
    #       the specific word embedding as output.
    model = LanguageModel(vocab_size, hy.embed_size, hy.window_size, hy.hidden_size, device, None)
    model.load_state_dict(torch.load("saved_runs/bert_transformer_0.pt"))

    window_embeddings = []

    model.eval()
    for window_text in chosen_windows:
        sequence = [vocab[word] for word in window_text]
        sequence = torch.tensor(sequence, dtype=torch.long, device=device)
        with torch.no_grad():
            _, output_embeddings = model(sequence)
            output_embeddings = np.copy(output_embeddings.cpu().numpy())
            window_embeddings.append(output_embeddings)

    embeddings = np.zeros((len(window_embeddings), hy.embed_size))
    window_embeddings = np.array(window_embeddings)
    subset_windows = []
    for i, word_id in enumerate(word_indices):
        word_embedding = window_embeddings[i, word_id, :]
        embeddings[i, :] = word_embedding
        subset_windows.append(chosen_windows[i][word_id - 5:word_id + 5])
    print("Created embeddings of shape ", embeddings.shape)

    # TODO: Use the plot_embeddings function above to plot the sentence
    #       and embeddings in two-dimensional space.
    plot_embeddings(subset_windows[:20], embeddings[:20, :], show_plot=False, plot_name="{}.png".format(word_set[0]))


if __name__ == '__main__':
    train_corpus = sys.argv[1]
    test_corpus = sys.argv[2]

    # Grab everything needed to create LanguageModel
    hy = Hyperparameters()
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("reverse_vocab.pkl", "rb") as f:
        reverse_vocab = pickle.load(f)
    vocab_size = len(vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_sentences = get_sentences(train_corpus)
    validation_sentences = get_sentences(test_corpus)
    sentences = training_sentences + validation_sentences
    dataset = LMDataset(sentences, vocab, reverse_vocab, hy.window_size)

    # TODO: Find all instances of sentences that have the words that
    #       vary by context.
    create_embeddings(["state", "states", "stated"])
    create_embeddings(["figure", "figures", "figured"])
    create_embeddings(["suspect"])
