package com.minilm.lite.utils;

/**
 * Utility class for calculating the cosine similarity between two vectors.
 * Cosine similarity measures the cosine of the angle between two non-zero vectors
 * in a multi-dimensional space, indicating how similar they are.
 */
public class CosineSimilarityUtil {

    private CosineSimilarityUtil() {
        throw new IllegalStateException("CosineSimilarityUtil class");
    }

    /**
     * Calculates the cosine similarity between two vectors.
     * The cosine similarity is defined as the dot product of the two vectors
     * divided by the product of their magnitudes (norms).
     *
     * @param vec1 The first vector, represented as an array of doubles.
     * @param vec2 The second vector, represented as an array of doubles.
     * @return A double value representing the cosine similarity between the two vectors.
     * The result ranges from -1 (completely opposite) to 1 (completely identical).
     * @throws IllegalArgumentException If the input vectors are not of the same length.
     */
    public static double calculate(double[] vec1, double[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vectors must be of the same length.");
        }

        double dotProduct = 0.0;
        double normVec1 = 0.0;
        double normVec2 = 0.0;

        // Compute the dot product and the norms of each vector
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            normVec1 += vec1[i] * vec1[i];
            normVec2 += vec2[i] * vec2[i];
        }

        // Calculate the cosine similarity
        return dotProduct / (Math.sqrt(normVec1) * Math.sqrt(normVec2));
    }
}