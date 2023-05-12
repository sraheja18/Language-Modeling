#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys
from nltk import bigrams, trigrams

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)


class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            if s:
                self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        """Calculates probability"""
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            if len(s) >= 2:
                num_words += len(s) + 1
                sum_logprob += self.logprob_sentence(s)
        return -(1.0 / num_words) * (sum_logprob)

    def logprob_sentence(self, sentence):
        """Calculates log probability of the sentence"""
        p = 0
        tri = list(trigrams(sentence))
        for x in tri:
            p += self.cond_logprob(x)
        if len(sentence) >= 2:
            p += self.cond_logprob(("start_of_sentence", sentence[0], sentence[1]))
            p += self.cond_logprob((sentence[-2], sentence[-1], "END_OF_SENTENCE"))
        return p

    # required, update the model when a sentence is observed
    # def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    # def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    # def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    # def vocab(self): pass


class Unigram(LangModel):
    def __init__(self, backoff=0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        """Add word to the model dictionary"""
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        """Adds the words of a sentence in model dictionary"""
        for w in sentence:
            self.inc_word(w)
        self.inc_word("END_OF_SENTENCE")

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()


class Trigram(LangModel):
    def __init__(self, backoff=0.000001, lambda1=0.4, lambda2=0.4):
        self.model_tri = dict()
        self.model_bi = dict()
        self.model_uni = dict()
        self.backoff = backoff
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = 1 - (lambda1 + lambda2)

    def inc_word_uni(self, w):
        """Adds word to the unigram model"""
        if w in self.model_uni:
            self.model_uni[w] += 1.0
        else:
            self.model_uni[w] = 1.0

    def inc_word_bi(self, w):
        """Adds word to the bi-gram model"""
        if w in self.model_bi:
            self.model_bi[w] += 1.0
        else:
            self.model_bi[w] = 1.0

    def inc_word_tri(self, w):
        """Adds word to the tri-gram model"""
        if w in self.model_tri:
            self.model_tri[w] += 1.0
        else:
            self.model_tri[w] = 1.0

    def fit_sentence(self, sentence):
        """Adds sentence to uni, bi and tri-gram model"""
        bi = list(bigrams(sentence))
        tri = list(trigrams(sentence))
        for w in sentence:
            self.inc_word_uni(w)
        for w in bi:
            self.inc_word_bi(w)
        for w in tri:
            self.inc_word_tri(w)
        if len(sentence) >= 2:
            self.inc_word_tri(("start_of_sentence", sentence[0], sentence[1]))
            self.inc_word_tri((sentence[-2], sentence[-1], "END_OF_SENTENCE"))
        self.inc_word_uni("start_of_sentence")
        self.inc_word_uni("END_OF_SENTENCE")
        self.inc_word_bi(("start_of_sentence", sentence[0]))
        self.inc_word_bi((sentence[-1], "END_OF_SENTENCE"))

    def norm(self):
        """Normalize and convert to log2-probs."""

        for w in self.model_tri:
            self.model_tri[w] /= self.model_bi[(w[0], w[1])]

        for w in self.model_bi:
            self.model_bi[w] /= self.model_uni[w[0]]

        s = sum(self.model_uni.values())

        for w in self.model_uni:
            self.model_uni[w] /= s

    def cond_logprob(self, word):
        """Calculates conditional log probability"""
        if word in self.model_tri:
            p = (
                self.lambda1 * self.model_tri[word]
                + self.lambda2 * self.model_bi[(word[1], word[2])]
                + self.lambda3 * self.model_uni[word[2]]
            )
        else:
            if (word[1], word[2]) in self.model_bi:
                p = (
                    self.lambda2 * self.model_bi[(word[1], word[2])]
                    + self.lambda3 * self.model_uni[word[2]]
                )
            elif word[2] in self.model_uni:
                p = self.lambda3 * self.model_uni[word[2]]
            else:
                p = self.backoff
        return log(p, 2)

    def vocab(self):
        """Dictionary of words"""
        return self.model_uni.keys()
