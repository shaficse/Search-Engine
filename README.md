
# A1-Search Engine

## Table of Contents
- [Overview](#overview)
- [Task 1 Preparation and Training](#task-1-preparation-and-training)
  - [Foundational Research Papers](#11-foundational-research-papers)
  - [Word Embedding Models](#12-word-embedding-models)
  - [Data Source Acknowledgment](#13-data-source-acknowledgment)
  - [Technical Components](#14-technical-components)
  - [Getting Started](#15-getting-started)
- [Task 2 Model Comparison and Analysis](#task-2-model-comparison-and-analysis)
- [Task 3 Search Engine - Web Application Development](#task-3-search-engine---web-application-development)
- [Contributing & Support](#contributing--support)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Overview
The Search Engine Project harnesses the power of information retrieval and natural language processing (NLP) to construct a search engine based on word embedding models, complemented by a user-friendly web interface. 

It aims to deliver the most relevant 10 passages in response to user queries by measuring similarity from a given corpus.

The comprehensive project encompasses several key tasks, including 
   - Data preparation
   - Training of word embedding models
   - Model comparison and analysis,   
   - Development of a web-based search engine.

## Task 1 Preparation and Training

### 1.1. Foundational Research Papers
The word embedding models in this project draw upon insights from key research papers:

1. **Efficient Estimation of Word Representations in Vector Space**: Introduces innovative techniques for training high-quality word vectors and showcases the effectiveness of the skip-gram model. [Read the paper here](https://arxiv.org/pdf/1301.3781.pdf)

2. **GloVe: Global Vectors for Word Representation**: Presents the GloVe model, an unsupervised learning algorithm for generating word embeddings by aggregating global word-word co-occurrence statistics. [Read the paper here](https://aclanthology.org/D14-1162.pdf)

These methodologies and insights form the backbone of our word embedding models, ensuring a scientifically robust approach.

### 1.2. Word Embedding Models

1.2.1. **Word2Vec (Skip-gram Architecture)**
   - **File**: `Task-1-Word2Vec (Skipgram).ipynb`
   - **Description**: Utilizes skip-gram architecture for generating word embeddings, effectively capturing word context within a specific window in a corpus.

1.2.2. **Word2Vec with Negative Sampling (Skip-gram Architecture)**
   - **File**: `Task-1-Word2Vec (Skipgram)-with-neg-sampling.ipynb`
   - **Description**: Enhances the basic Word2Vec model by incorporating negative sampling, improving computational efficiency and performance, especially with large datasets.

1.2.3. **GloVe (Global Vectors for Word Representation)**
   - **File**: `Task-1-GloVe-from-Scratch.ipynb`
   - **Description**: Constructs word embeddings by analyzing global word-word co-occurrence statistics, balancing global statistics and local context.

### 1.3. Data Source Acknowledgment

#### Reuters Dataset
The project utilizes the Reuters-21578 Text Categorization Collection from the NLTK library. This extensive dataset provides a rich source for supervised learning in NLP. Special thanks to the creators and maintainers of the Reuters dataset and the NLTK library.

**Source Details:**
- **Dataset Name**: Reuters-21578 Text Categorization Collection
- **Provided By**: NLTK (Natural Language Toolkit)
- **Dataset Description**: A collection of categorized news stories, ideal for text classification and topic modeling.
- **License**: Public Domain (Users are requested to cite the source)
- **Source Link**: [NLTK Data](http://www.nltk.org/nltk_data/)

### 1.4. Technical Components
- **[PyTorch](https://pytorch.org/)**: Used for building and training the Word2Vec model.
- **[NLTK](https://www.nltk.org/)**: Utilized for natural language processing tasks like tokenization and accessing the Reuters dataset.
- **[Flask](https://flask.palletsprojects.com/en/2.0.x/)**: Serves the backend application and handles API requests.
- **[HTML/CSS/JavaScript](https://developer.mozilla.org/en-US/docs/Web/HTML)**: Used for building the frontend interface.

### 1.5. Getting Started

#### Prerequisites
- Python 3.6 or higher
- PyTorch
- Flask
- NLTK
- scipy

#### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/shaficse/Search-Engine.git
   ```
2. Install the required Python packages:
   ```sh
   pip install -r all_requirements.txt
   ```

## Task-2 Model Comparison and Analysis

The models were trained on a selected subset of the Reuters corpus from NLTK, containing 500 passages out of a total of 54,716, and 2677 tokens out of 1720917. The training performance was assessed based on the average training loss and the total time taken for training

### 2.1. Training Loss & Time

Comparison of Skip-gram, Skip-gram with Negative Sampling, and GloVe models based on training loss and time, providing insights into learning efficiency and computational demands.

#### Training Loss
| Model                          | Average Training Loss |
|--------------------------------|-----------------------|
| Skip-gram                      | 8.133966            |
| Skip-gram with Negative Sampling | 1.977957             |
| GloVe Scratch                        | 0.724803 |

- **Training Loss:**
- Skip-gram: Exhibited an average training loss of 8.133966, indicating room for optimization.
- Skip-gram with Negative Sampling: Showed improved efficiency with a lower average training loss of 1.977957.
- GloVe Scratch: Achieved the lowest training loss of 0.724803, indicating a more effective learning during training.

#### Training Time

| Model                          | Total Training Time |
|--------------------------------|---------------------|
| Skip-gram                      | 18m 4s            |
| Skip-gram with Negative Sampling | 17m 8s             |
| GloVe Scratch                    | 1m 54s              |


- **Training Time:**
- Skip-gram: Took 18 minutes and 4 seconds for training.
- Skip-gram with Negative Sampling: Required 17 minutes and 8 seconds, slightly faster than Skip-gram.
- GloVe Scratch: Was significantly faster with a training time of 1 minute and 54 seconds.

This comparison sheds light on the trade-offs between models in terms of learning efficiency and computational needs.

### 2.2. Model's Accuracy on Analogy Dataset

Comparison of various models on the   `capital-common-countries (semantic)` and `past-tense (syntactic)` [analogy dataset](https://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt) reveals significant insights:
| Model                             | Window Size | Training Loss(taken from Traning Notebooks) | Syntactic Accuracy | Semantic Accuracy |
|-----------------------------------|-------------|---------------|--------------------|-------------------|
| Skip-gram                         | 2           | 8.133966         | 0.00%              | 0.00%             |
| Skip-gram with Negative Sampling  | 2           | 1.977957       | 0.00%              | 0.00%             |
| GloVe Scratch                    | 2           | 0.724803        | 0.00%              | 0.00%             |
| GloVe (Pre-trained Gensim)        | N/A         | N/A           | 53.40%             | 54.97%            |


- **Custom Models' Zero Accuracy:**
Skip-gram, Skip-gram with Negative Sampling,and Scratch GloVe models demonstrated zero accuracy, likely due to the **out-of-vocabulary issue**.

- **Strong Performance of Pre-trained GloVe:**
Over 50% accuracy in both tasks, highlighting the importance of extensive training and diverse datasets.


### 2.3. Spearman's Rank Correlation Summary
The Spearman's rank correlation analysis reveals varied performance among different word embedding models in aligning with human-perceived word similarities on [wordsim353_sim_rel data](http://alfonseca.org/eng/research/wordsim353.html).

| Model                     | Spearman's Rank Correlation |
|---------------------------|-----------------------------|
| Skipgram                  |  0.199                     |
| Skipgram-Neg-Sampling     | 0.043                    |
| GloVe Scratch             | -0.318                  |
| GloVe Gensim              | 0.602                      |


- Skipgram: shows a modest positive correlation (0.199), suggesting some alignment with human semantic understanding.

- Skipgram-Neg-Sampling: has a lower correlation (0.043), indicating a weaker alignment.

- GloVe Scratch: presents a negative correlation (-0.318), implying a divergence from human semantic patterns.

- GloVe Gensim: significantly outperforms the other models with a strong positive correlation (0.602), indicating a high level of alignment with human semantic understanding.

The pre-trained GloVe model from Gensim showcases superior performance, significantly aligning with human judgment in word similarities.


Future research directions may include enhancing custom-trained models by:
   - Expanding the training corpus to expose the model to a more varied text.

The details of the Comparision and Analysis are experiemented on this notebook `Task-2-Model-Comparison-Analysis.ipynb`.

## Task-3 Search Engine - Web Application Development

This Task introduces a web application designed to simplify the retrieval of information from a text corpus. Users can input search queries into a streamlined interface, which then utilizes one of our trained word embedding model to compute the similarity between the query and the corpus by computing the dot product of them. 

- **Model Selection**: 
Implement a function to compute the dot product between an input query and a corpus, retrieving the top 10 most similar passages with a focus on semantic similarity. it's crucial to choose the right model.


   - Based on the priority of semantic similarity in the task and considering Spearman's rank correlation, Skip-gram is recommended. 
   - Despite Skip-gram's longer training time and higher loss, its modest positive correlation with human judgment of word similarity suggests it may align slightly better with human semantic understanding.

- **Application Development**: Developed using Flask, a flexible micro web framework. Flask's simplicity and extensibility make it perfect for creating efficient, scalable web applications.

- **Launching the Application**:
   1. Start the Flask server:
      ```sh
      python app.py
      ```
   2. Open your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

   3. Input a search query and click "Search" or press 'Enter'.

   4. View the ranked 10  most of the similar passages to your query.

   <img src="screenshots/app-1.png" >

   For example in the above Screenshot-  The presence of other less relevant passages in the top results highlights the need for improved precision in the model's information retrieval capabilities.

   ***For Live Demo from Huggingface Space [https://huggingface.co/spaces/shaficse/a1-search](https://huggingface.co/spaces/shaficse/a1-search)***



## Contributing & Support
Contributions are welcome. For issues or questions, please open an issue in the repository.

## License
This project is licensed under the MIT License.[LICENSE](LICENSE)

## Acknowledgments
- Full acknowledgments to the authors of the foundational research papers.
- Gratitude to [Chaklam Silpasuwanchai](https://github.com/chaklam-silpasuwanchai) for providing valuable resources. The code used in this project was inspired by and adapted from his repository on [Python for Natural Language Processing](https://github.com/chaklam-silpasuwanchai/Python-fo-Natural-Language-Processing).

- The maintainers of the Reuters dataset and NLTK library.
