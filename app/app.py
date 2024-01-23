from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import torch.nn as nn
import json
import nltk
import string

# Define Flask application
app = Flask(__name__)

# Define the Skipgram model class
class Skipgram(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(Skipgram, self).__init__()
        # Embedding layers for center and outside words
        self.embedding_center = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
    
    def forward(self, center, outside, all_vocabs):
        # Obtain embeddings for center, outside, and all vocabulary words
        center_embedding = self.embedding_center(center)
        outside_embedding = self.embedding_outside(outside)
        all_vocabs_embedding = self.embedding_outside(all_vocabs)
        
        # Calculate top and lower terms for loss computation
        top_term = torch.exp(outside_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2))
        lower_term = all_vocabs_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2)
        lower_term_sum = torch.sum(torch.exp(lower_term), 1)
        
        # Calculate and return loss
        loss = -torch.mean(torch.log(top_term / lower_term_sum))
        return loss
        
# Load model configuration
with open('word2vec_config.json', 'r') as config_file:
    config = json.load(config_file)

# Initialize model with loaded configuration
model = Skipgram(voc_size=config['voc_size'], emb_size=config['emb_size'])

# Load model state
model_path = 'word2vec_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Load word2index, index2word, and corpus
word2index_path = 'word2index.json'
index2word_path = 'index2word.json'
corpus_path = 'corpus.txt'

with open(word2index_path, 'r') as file:
    word2index = json.load(file)

with open(index2word_path, 'r') as file:
    index2word = json.load(file)


def load_corpus(file_path):
    corpus = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip removes leading/trailing whitespace
            line = line.strip()
            # print(line)
            if line:  # Add non-empty lines to the corpus
                corpus.append(line)
    return corpus

# Load the corpus
corpus = load_corpus(corpus_path)

def preprocess(text):
    # Tokenizes the text into words and converts all characters to lowercase
    tokens = nltk.word_tokenize(text.lower())
    return tokens

def get_embedding(text, model, word2index):
    """
    Converts a text input to its corresponding average embedding.
    """
    tokens = preprocess(text)  # Preprocess the text to get tokens
    embeddings = []
    for token in tokens:
        index = word2index.get(token, word2index.get('<UNK>'))
        word_tensor = torch.LongTensor([index])

        if index >= 0 and index < model.embedding_center.weight.shape[0]:
            embed_center = model.embedding_center(word_tensor)
            embed_outside = model.embedding_outside(word_tensor)
            embed = (embed_center + embed_outside) / 2
            embeddings.append(embed.detach().numpy())
        else:
            embeddings.append(np.zeros(model.embedding_center.weight.shape[1]))
    
    # Average the embeddings
    if embeddings:
        embeddings = np.array(embeddings)
        text_embedding = np.mean(embeddings, axis=0)
    else:
        text_embedding = np.zeros(model.embedding_center.weight.shape[1])
    
    # Make sure the embedding is a 1-D array
    text_embedding = text_embedding.flatten()  # Flatten the array to ensure it's 1-D
    
    return text_embedding


def retrieve_top_passages(query, corpus, model, word2index, top_n=10):
    """
    Computes the dot product between the input query and each passage in the corpus,
    and retrieves the top N most similar passages.
    """
    query_embedding = get_embedding(query, model, word2index)
    similarities = []

    for passage in corpus:
        passage_embedding = get_embedding(passage, model, word2index)
        similarity = np.dot(query_embedding, passage_embedding)
        similarities.append(similarity)

    
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    top_indices = sorted_indices[:top_n]
    
    # Normalize the scores to be percentages of the max score
    # max_score = max([similarities[idx] for idx in top_indices])
    # top_passages = [(corpus[idx], (similarities[idx] / max_score) * 100) for idx in top_indices]
 
    top_passages = [(corpus[idx], (similarities[idx]) * 100) for idx in top_indices]
    
    return top_passages


# Serve the HTML page at the root
@app.route('/')
def index():
    return render_template('index.html')

# Handle search requests

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    search_text = data['search_text']
    
    top_passages = retrieve_top_passages(search_text, corpus, model, word2index, top_n=10)
    
    # Convert results to a format that can be JSON-serialized, including percentage scores
    results = [
        {
            'rank': rank+1,
            'sentence': sentence,
            'score': round(score, 2) if score > 0 else 0,
        } for rank, (sentence, score) in enumerate(top_passages)
    ]
    
    return jsonify(results)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
