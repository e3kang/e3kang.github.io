import numpy as np
from collections import Counter

DATA_FILE = 'all_sentiment_shuffled.txt'
TRAIN_FRACTION = 0.80
ALPHA = 1.0


'''
Read data from file into a Python array.

Each line in the .txt file is: topic, polarity (its label), doc id, tokens

Output: (docs, labels) both arrays
'''
def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels


'''
Split docs and labels into a training and test set.
'''
def split_data(docs, labels, train_fraction):
    # no shuffle needed
    split_point = int(train_fraction * len(docs))
    return (docs[:split_point], labels[:split_point], docs[split_point:], labels[split_point:])



'''
Count how often each word occurs in the documents of each class.
'''
def count_words_per_label(documents, labels):
    word_counts = {}
    for doc, label in zip(documents, labels):
        word_counts.setdefault(label, Counter()).update(doc)
    return word_counts


'''
Log prior of labels.
'''
def compute_log_priors(labels):
    label_counts = Counter(labels)
    total = len(labels)
    return {label: np.log(count / total) for label, count in label_counts.items()}

'''
Compute "smoothed" log likelihood of each vocab word, given a class.
'''
def compute_log_likelihoods(counts, vocabulary, alpha):
    # add alpha smoothing:
    denominator = sum(counts.values()) + alpha * len(vocabulary)
    return {w: np.log((counts[w] + alpha) / denominator) for w in vocabulary}


'''
Naive Bayes parameters (log priors and likelihoods) from training data.
'''
def train_nb(documents, labels, alpha=ALPHA):
    word_counts = count_words_per_label(documents, labels)
    vocabulary = set(w for counts in word_counts.values() for w in counts)
    return {
        'log_priors': compute_log_priors(labels),
        'log_likelihoods': {label: compute_log_likelihoods(counts, vocabulary, alpha) for label, counts in word_counts.items()},
        'vocabulary': vocabulary,
    }

'''
The score of the label (log probability)
'''
def score_doc_label(document, label, model):
    log_likelihoods = model['log_likelihoods'][label]
    score = model['log_priors'][label]
    for w in document:
        # Words never seen in training are skipped, not modelled
        if w in model['vocabulary']:
            score += log_likelihoods[w]
    return score


'''
Scores of a label under each class
'''
def score_all_labels(document, model):
    return {label: score_doc_label(document, label, model) for label in model['log_priors']}


'''
Return class with highest likelihood.
'''
def classify_nb(document, model):
    scores = score_all_labels(document, model)
    return max(scores, key=scores.get)


'''
Predicted labels for a whole class of documents.
'''
def classify_documents(docs, model):
    return [classify_nb(doc, model) for doc in docs]

'''
Return accuracy of labelling: correct labels / # of data points
'''
# Fraction of documents whose guessed label matches the true label.
def accuracy(true_labels, guessed_labels):
    correct = sum(1 for t, g in zip(true_labels, guessed_labels) if t == g)
    return correct / len(true_labels)


# Posterior P(label | document) for every class (normalised from the joint log scores)
def posterior_probabilities(document, model):
    scores = score_all_labels(document, model)
    # Subtracting the max prevents underflow
    top = max(scores.values())
    unnormalised = {label: np.exp(s - top) for label, s in scores.items()}
    total = sum(unnormalised.values())
    return {label: p / total for label, p in unnormalised.items()}


# Show misclassified documents. Sorted by most confidently wrong
def find_misclassified(docs, true_labels, model, max_length):
    errors = []
    for doc, true_label in zip(docs, true_labels):
        # Short documents only (for human readability)
        if len(doc) > max_length:
            continue
        posteriors = posterior_probabilities(doc, model)
        guess = max(posteriors, key=posteriors.get)
        if guess != true_label:
            errors.append((posteriors[guess], true_label, guess, doc))
    errors.sort(key=lambda e: e[0], reverse=True)
    return errors


'''
Words in a document that pulled hardest towards mistakes.
'''
def strongest_evidence(doc, guess, true_label, model, k):
    likelihoods = model['log_likelihoods']
    ratios = {w: likelihoods[guess][w] - likelihoods[true_label][w] for w in set(doc) if w in model['vocabulary']}
    return sorted(ratios, key=ratios.get, reverse=True)[:k]



def run_sanity_checks(model):
    print('--- Sanity checks ---')
    for word in ['great', 'bad']:
        for label in sorted(model['log_priors']):
            prob = np.exp(score_doc_label([word], label, model))
            print(f'P(doc=[{word!r}], {label}) = {prob:.6f}')
    unseen_doc = ['a', 'top-quality', 'performance']
    for label in sorted(model['log_priors']):
        print(f'log P({unseen_doc}, {label}) = {score_doc_label(unseen_doc, label, model):.4f}')
    for doc in [['great'], ['bad'], ['i', 'love', 'it'], ['waste', 'of', 'money'], ['highly', 'recommend'], ['very', 'disappointed']]:
        print(f'classify_nb({doc}) = {classify_nb(doc, model)}')


'''
Test accuracy.
'''
def run_evaluation(eval_docs, eval_labels, model):
    print('--- Evaluation ---')
    guesses = classify_documents(eval_docs, model)
    print(f'Evaluation documents: {len(eval_docs)}')
    print(f'Accuracy: {accuracy(eval_labels, guesses):.4f}')


# Print the most confidently misclassified short documents.
def run_error_analysis(eval_docs, eval_labels, model, max_length=60, how_many=5):
    print('--- Error analysis ---')
    errors = find_misclassified(eval_docs, eval_labels, model, max_length)
    print(f'Misclassified documents with at most {max_length} tokens: {len(errors)}')
    for confidence, true_label, guess, doc in errors[:how_many]:
        print(f'\ntrue={true_label} guessed={guess} P(guess|doc)={confidence:.4f}')
        print('text:', ' '.join(doc))
        print('strongest words for the wrong class:', strongest_evidence(doc, guess, true_label, model, 8))


if __name__ == '__main__':
    all_docs, all_labels = read_documents(DATA_FILE)
    train_docs, train_labels, eval_docs, eval_labels = split_data(all_docs, all_labels, TRAIN_FRACTION)
    print(f'Training documents: {len(train_docs)}')

    model = train_nb(train_docs, train_labels)
    print(f'Vocabulary size: {len(model["vocabulary"])}')

    run_sanity_checks(model)
    run_evaluation(eval_docs, eval_labels, model)
    run_error_analysis(eval_docs, eval_labels, model)
