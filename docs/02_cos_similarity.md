# What is Cosine Similarity?

Cosine similarity is a metric used to measure how similar two vectors are, focusing on their direction rather than their magnitude. This concept is particularly useful in fields like information retrieval, text analysis, and machine learning, where it can help in comparing documents, images, or patterns by quantifying their similarity in terms of orientation in a multi-dimensional space.

### How Cosine Similarity Works

The cosine similarity between two vectors $$A$$ and $$B$$ is calculated by taking the dot product of the vectors and dividing it by the product of their magnitudes (or lengths). Mathematically, it is represented as:

$$
\text{similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\|\|B\|}
$$

where:

- $$\theta$$ is the angle between the vectors $$A$$ and $$B$$,
- $$A \cdot B$$ is the dot product of vectors $$A$$ and $$B$$,
- $$\|A\|$$ and $$\|B\|$$ are the magnitudes (or Euclidean norms) of vectors $$A$$ and $$B$$, respectively.

The dot product $$A \cdot B$$ is calculated as $$A^T B= \sum_{i =1} ^{n} A_iB_i$$, which is the sum of the products of the corresponding entries of the vectors. The magnitude of a vector $$A$$, denoted as $$\|A\|$$, is calculated as $$\sqrt{A_1^2 + A_2^2 + \ldots + A_n^2}$$.

### Numerical Example

Consider two vectors $$D1 = [1][1][1][1][1]$$ and $$D2 = [1][1][1][1]$$. To calculate their cosine similarity:

1. Calculate the dot product $$D1 \cdot D2 = 1\times0+1\times0+1\times1+1\times1+1\times0+0\times1+0\times1=2$$.
2. Calculate the magnitudes $$\|D1\| = \sqrt{1^2+1^2+1^2+1^2+1^2+0^2+0^2}=\sqrt{5}$$ and $$\|D2\| = \sqrt{0^2+0^2+1^2+1^2+0^2+1^2+1^2}=\sqrt{4}$$.
3. Finally, calculate the cosine similarity: $$\text{similarity}(D1, D2) = \frac{2}{\sqrt{5} \sqrt{4}} = \frac{2}{\sqrt{20}} = 0.44721$$.

### Interpretation

- A cosine similarity of 1 means the vectors are identical in orientation.
- A cosine similarity of 0 indicates orthogonality (no similarity).
- A cosine similarity of -1 implies diametrically opposed vectors.

### Applications

Cosine similarity is widely used in various applications, including but not limited to:

- **Text Analysis**: Comparing documents or sentences by converting text to vectors (e.g., TF-IDF vectors) and measuring their cosine similarity.
- **Recommendation Systems**: Identifying items similar to a user's preferences by comparing feature vectors.
- **Image and Video Analysis**: Comparing visual content by treating image or video features as vectors.

### Implementation

Cosine similarity can be easily implemented using popular Python libraries such as `sklearn.metrics.pairwise.cosine_similarity` for arrays or matrices, and `SciPy` library's cosine distance function for more general cases[1].

This metric's simplicity and effectiveness in capturing the orientation similarity between vectors make it a go-to choice for comparing multidimensional data across various domains[1][2][3].

Citations:
[1] https://www.learndatasci.com/glossary/cosine-similarity/
[2] https://www.geeksforgeeks.org/cosine-similarity/
[3] https://builtin.com/machine-learning/cosine-similarity
[4] https://www.youtube.com/watch?v=e9U0QAFbfLI&vl=en
[5] https://www.machinelearningplus.com/nlp/cosine-similarity/
[6] https://www.sciencedirect.com/topics/computer-science/cosine-similarity
[7] https://www.datastax.com/guides/what-is-cosine-similarity
[8] https://towardsdatascience.com/cosine-similarity-how-does-it-measure-the-similarity-maths-behind-and-usage-in-python-50ad30aad7db
