# 🎬 Sentiment Analysis using LSTM and BiLSTM

A deep learning project for classifying movie reviews as **Positive** or **Negative** using LSTM and Bidirectional LSTM (BiLSTM) networks in PyTorch.

The project uses the **IMDb Dataset of 50K Movie Reviews** and demonstrates the complete NLP pipeline, from raw text preprocessing to model evaluation and prediction on new and ambiguous reviews.

---

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to determine the emotional polarity of a piece of text.

In this project, movie reviews are classified into two categories:

- 🟢 **Positive**
- 🔴 **Negative**

Two recurrent neural network architectures were implemented and compared:

1. **LSTM (Long Short-Term Memory)**
2. **BiLSTM (Bidirectional Long Short-Term Memory)**

The project includes:

- Text preprocessing
- Tokenization
- Vocabulary creation
- Word-to-index mapping
- Padding and truncation
- Packed sequences
- PyTorch Dataset and DataLoader
- Word embeddings
- LSTM
- BiLSTM
- Dropout
- Model training
- Validation
- Test evaluation
- Confusion matrices
- Error analysis
- New review prediction
- Ambiguous review testing
- Saving trained model artifacts

---

## 📂 Dataset

The project uses the **IMDb Dataset of 50K Movie Reviews**.

The dataset contains:

- **50,000 movie reviews**
- **25,000 positive reviews**
- **25,000 negative reviews**

### Dataset Split

| Dataset | Samples |
|---|---:|
| Training | 40,000 |
| Validation | 5,000 |
| Testing | 5,000 |
| **Total** | **50,000** |

The dataset is balanced, with an equal number of positive and negative reviews.

---

## 🔄 NLP Pipeline

The complete preprocessing and modeling pipeline is:

Raw Movie Review
       ↓
Text Preprocessing
       ↓
Tokenization
       ↓
Vocabulary Creation
       ↓
Convert Words → Integer IDs
       ↓
Padding / Truncation
       ↓
PyTorch Dataset
       ↓
DataLoader
       ↓
Embedding Layer
       ↓
LSTM / BiLSTM
       ↓
Fully Connected Layer
       ↓
Sentiment Prediction
       ↓
Positive / Negative

🧹 Data Preprocessing

The raw movie reviews are first cleaned and tokenized.

1. Text Cleaning

HTML tags such as <br /> are removed and unnecessary whitespace is cleaned.

2. Tokenization

Reviews are converted into lowercase tokens using whitespace-based tokenization.

For example:

"This movie was absolutely amazing!"

becomes:

["this", "movie", "was", "absolutely", "amazing!"]
3. Vocabulary Creation

A vocabulary is created using the training dataset.

The vocabulary contains the most frequent 20,000 words.

Two special tokens are added:

<PAD> → 0
<UNK> → 1

Therefore, the total vocabulary size is:

20,002
4. Word-to-Integer Conversion

Each word is converted into its corresponding integer ID.

Words that are not present in the vocabulary are mapped to:

<UNK> → 1
5. Padding and Truncation

The maximum sequence length is set to:

500

Reviews longer than 500 tokens are truncated, while shorter reviews are padded using:

<PAD> → 0

This ensures that every review has the same sequence length and can be processed efficiently in batches.

6. Packed Sequences

The original sequence lengths are preserved before padding.

pack_padded_sequence is used in the LSTM and BiLSTM models so that the recurrent networks do not unnecessarily process padding tokens.

This allows the models to focus on the actual review content rather than the padded positions.

🧠 Model 1 — LSTM
What is LSTM?

LSTM stands for Long Short-Term Memory.

It is a type of recurrent neural network designed to process sequential data while retaining important information over longer sequences.

An LSTM processes the review from beginning to end:

Word 1 → Word 2 → Word 3 → ... → Word N

At each step, the LSTM maintains:

Hidden state
Cell state

These states allow the network to retain information from earlier words.

LSTM Architecture
Input Sequence
     ↓
Embedding Layer
     ↓
LSTM
     ↓
Final Hidden State
     ↓
Fully Connected Layer
     ↓
Sentiment
Configuration
Parameter	Value
Vocabulary Size	20,002
Embedding Dimension	128
Hidden Size	128
Sequence Length	500
Number of LSTM Layers	1
Direction	Unidirectional
Loss Function	BCEWithLogitsLoss
Optimizer	Adam
Learning Rate	0.001
LSTM Data Flow
Input
[Batch Size, 500]

        ↓

Embedding

[Batch Size, 500, 128]

        ↓

LSTM

[1, Batch Size, 128]

        ↓

Final Hidden State

[Batch Size, 128]

        ↓

Fully Connected Layer

[Batch Size, 1]

        ↓

Sentiment
🔄 Model 2 — BiLSTM
What is BiLSTM?

BiLSTM stands for Bidirectional Long Short-Term Memory.

Unlike a standard LSTM, a BiLSTM processes the sequence in two directions.

Forward Direction
Word 1 → Word 2 → Word 3 → ... → Word N
Backward Direction
Word N → Word N-1 → Word N-2 → ... → Word 1

The information from both directions is combined.

This allows the model to use both:

Previous context
Future context

when representing the sequence.

BiLSTM Architecture
                 ┌──→ Forward LSTM ───┐
Input Sequence ──┤                    ├──→ Concatenate
                 └──→ Backward LSTM ──┘
                                           ↓
                                        Dropout
                                           ↓
                                  Fully Connected Layer
                                           ↓
                                      Sentiment
Configuration
Parameter	Value
Vocabulary Size	20,002
Embedding Dimension	128
Hidden Size	128 per direction
Sequence Length	500
Number of LSTM Layers	1
Direction	Bidirectional
Dropout	0.5
Loss Function	BCEWithLogitsLoss
Optimizer	Adam
Learning Rate	0.001

Because the BiLSTM has two directions:

128 forward features
+
128 backward features
=
256 features

These 256 features are passed through dropout and then into the final classification layer.

🆚 LSTM vs BiLSTM
Feature	LSTM	BiLSTM
Processing Direction	Forward	Forward + Backward
Context	Previous context	Previous + Future context
Hidden Representation	128 features	256 features
Dropout	Not used	0.5
Computational Cost	Lower	Higher
Architecture	Simpler	More complex
Information Flow	One direction	Two directions
Example

Consider the sentence:

The movie was not very good.

A standard LSTM reads:

The → movie → was → not → very → good

A BiLSTM processes the sentence in both directions, allowing the representation to incorporate information from both sides of the sequence.

This can be useful when the meaning of a word depends strongly on surrounding context.

📊 Model Results

Both models were trained and evaluated on the same IMDb dataset split.

LSTM
Best Validation Accuracy: 87.96%
Test Accuracy: 87.94%
BiLSTM
Best Validation Accuracy: 86.72%
Test Accuracy: 86.46%
Comparison
Model	Best Validation Accuracy	Test Accuracy
LSTM	87.96%	87.94%
BiLSTM	86.72%	86.46%

In this particular experiment, the LSTM achieved a slightly higher test accuracy than the BiLSTM.

This does not mean that LSTM is generally better than BiLSTM. The performance depends on factors such as the dataset, preprocessing, architecture, hyperparameters, and training configuration.

📈 Model Evaluation

The notebook contains several evaluation methods.

1. Training and Validation Loss

Training and validation loss curves are plotted to observe how the models learn over epochs.

These plots also help identify possible overfitting.

2. Training and Validation Accuracy

Training and validation accuracy are plotted to compare model performance during training.

3. Confusion Matrix

Confusion matrices are used to understand the classification errors made by each model.

The confusion matrix contains:

True Negative
False Positive
False Negative
True Positive
4. Test Set Evaluation

The final models are evaluated on the unseen test set.

5. New Review Prediction

The trained models are also tested using new movie reviews.

Example:

Review:
This movie was absolutely amazing. The acting was excellent and I loved every minute of it.

Prediction:
Positive

Another example:

Review:
The movie was boring and predictable. I did not enjoy it at all.

Prediction:
Negative
6. Ambiguous Review Testing

The models were also tested on difficult reviews containing mixed or conflicting sentiments.

This helps demonstrate how recurrent models handle more complicated language and provides examples for error analysis.

🔍 Error Analysis

Testing the models with ambiguous reviews showed that some sentences are more difficult to classify.

For example, a review may contain many negative expressions before ending with a positive statement.

A recurrent model may focus strongly on the negative context and classify the review incorrectly.

This demonstrates an important limitation of sentiment classification:

Correctly understanding sentiment often requires understanding the relationship between words across the entire sentence.

The models were therefore tested with ambiguous and mixed-sentiment reviews to examine how they handle difficult cases.

These experiments helped identify cases where the models were confident but incorrect, as well as cases where the sentiment was difficult to determine even from a human perspective.

💾 Saved Model Files

The trained model artifacts are included in the repository so that the models can be reused without retraining.

models/
│
├── LSTM/
│   ├── model.pth
│   ├── word_to_int.pkl
│   └── config.pkl
│
└── BiLSTM/
    ├── model.pth
    ├── word_to_int.pkl
    └── config.pkl
LSTM

The LSTM directory contains the trained LSTM model and the preprocessing/configuration files required to reuse the model.

model.pth

Contains the trained LSTM model weights.

word_to_int.pkl

Contains the vocabulary mapping used during preprocessing.

It maps words to their corresponding integer IDs.

config.pkl

Contains the model configuration required to recreate the LSTM architecture.

The configuration includes parameters such as:

Vocabulary size
Embedding dimension
Hidden size
Maximum sequence length
BiLSTM

The BiLSTM directory contains the equivalent trained model artifacts for the Bidirectional LSTM.

model.pth

Contains the trained BiLSTM model weights.

word_to_int.pkl

Contains the vocabulary mapping used during preprocessing.

config.pkl

Contains the model configuration required to recreate the BiLSTM architecture.

The configuration includes parameters such as:

Vocabulary size
Embedding dimension
Hidden size
Dropout
Maximum sequence length
Bidirectional setting
📁 Project Structure
Sentiment-Analysis/
│
├── Sentiment_Analysis_LSTM_BiLSTM.ipynb
│
├── models/
│   │
│   ├── LSTM/
│   │   ├── model.pth
│   │   ├── word_to_int.pkl
│   │   └── config.pkl
│   │
│   └── BiLSTM/
│       ├── model.pth
│       ├── word_to_int.pkl
│       └── config.pkl
│
├── requirements.txt
│
├── LICENSE
│
└── README.md
🛠️ Technologies Used
Python
PyTorch
Pandas
NumPy
Scikit-learn
Matplotlib
Natural Language Processing (NLP)
LSTM
BiLSTM
IMDb Dataset
🚀 Future Work

The next stage of this project is to compare the recurrent neural network models with a pretrained Transformer-based model.

The planned model is:

DistilBERT

The project will therefore progress from recurrent neural networks to Transformer-based NLP:

LSTM
   ↓
BiLSTM
   ↓
DistilBERT

The DistilBERT model will use the Hugging Face Transformers library and will be fine-tuned for binary sentiment classification.

This comparison will help demonstrate the differences between recurrent neural network architectures and modern Transformer-based language models.

🎯 Learning Outcomes

Through this project, I learned and implemented:

Text preprocessing
Tokenization
Vocabulary creation
Word-to-index mapping
Handling unknown words
Padding and truncation
Sequence length analysis
PyTorch Dataset and DataLoader
Word embeddings
LSTM networks
Bidirectional LSTM networks
Packed sequences
Dropout
Binary sentiment classification
Model training
Validation
Test evaluation
Confusion matrices
Overfitting analysis
Error analysis
Saving trained PyTorch models
📜 License

This project is licensed under the MIT License.

See the LICENSE file for more information.

👩‍💻 Author

Vashundthera

This project was created as part of my learning journey in:

Natural Language Processing
Deep Learning
PyTorch
AI Engineering
