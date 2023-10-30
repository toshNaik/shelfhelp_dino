import numpy as np
from scipy.spatial import distance
from scipy.stats import chisquare
from numpy.linalg import norm

# Given distributions
P = np.array([0.4, 0.2, 0.2, 0.15, 0.05])
Q = np.array([0.35, 0.25, 0.15, 0.15, 0.1])


# This metric evaluates the cosine of the angle between two vectors, 
# yielding a similarity measure. A cosine value of 1 means the vectors 
# are identical, and a value of 0 means the vectors are orthogonal (completely dissimilar).
def cosine_similarity(p, q):
    return np.dot(p, q) / (norm(p) * norm(q))


# JSD is a method of measuring the similarity between two probability distributions. 
# It's derived from the Kullback-Leibler divergence, but unlike KL-divergence, 
# JSD is symmetric and always has a finite value. The square root of JSD gives you 
# the Jensen-Shannon distance, which is a proper metric.
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


# This statistical test determines if there's a significant difference between the expected 
# frequencies (one scene's composition) and the observed frequencies (other scene's composition). 
# A low Chi-squared value indicates that the scenes are similar in composition.
def chi_squared(p, q):
    return chisquare(p, q).statistic


# This is a measure of how one probability distribution differs from a second, reference 
# probability distribution. It's not symmetric, meaning KL(P||Q) is not the same as KL(Q||P).
def kullback_leibler_divergence(p, q):
    return np.sum(p * np.log(p / q))


# This is a type of f-divergence and can be used to measure the similarity between two 
# probability distributions. It provides a value between 0 (identical distributions) and 
# 1 (maximally different distributions).
def hellinger_distance(p, q):
    return 1.0 / np.sqrt(2) * norm(np.sqrt(p) - np.sqrt(q))


# This is another way to measure the difference between two probability distributions. 
# It captures the maximal difference between the probabilities of events under two distributions.
def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))


# Calculate and print metrics
print("Euclidean Distance:", distance.euclidean(P, Q))
print("Cosine Similarity:", cosine_similarity(P, Q))
print("Jensen-Shannon Divergence:", jensen_shannon_divergence(P, Q))
print("Chi-Squared:", chi_squared(P, Q))
print("Kullback-Leibler Divergence:", kullback_leibler_divergence(P, Q))
print("Hellinger Distance:", hellinger_distance(P, Q))
print("Total Variation Distance:", total_variation_distance(P, Q))