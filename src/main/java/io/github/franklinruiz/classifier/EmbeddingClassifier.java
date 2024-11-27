package io.github.franklinruiz.classifier;

import io.github.franklinruiz.encoder.MiniLMEmbedder;
import io.github.franklinruiz.utils.ScoredLabel;

import java.util.*;

/**
 * EmbeddingClassifier is a generic text classifier that uses embeddings to categorize input text.
 * It leverages a MiniLMEmbedder model to generate embeddings for both the input text and labeled examples.
 * Classification is performed by comparing the embeddings using cosine similarity.
 *
 * @param <L> The type of label used for classification (e.g., String, Enum, etc.).
 */
public class EmbeddingClassifier<L> implements TextClassifier<L> {

    private final MiniLMEmbedder embeddingModel;  // MiniLMEmbedder for generating text embeddings
    private final Map<L, List<double[]>> exampleEmbeddingsByLabel; // Pre-computed embeddings for labeled examples
    private final int maxResults; // Maximum number of results to return
    private final double minScore; // Minimum score threshold for classification
    private final double meanToMaxScoreRatio; // Weight ratio for mean vs. max score aggregation

    /**
     * Constructs an EmbeddingClassifier with default settings for maxResults, minScore, and meanToMaxScoreRatio.
     *
     * @param embeddingModel  The MiniLMEmbedder used to generate embeddings.
     * @param examplesByLabel A map of labels to lists of example texts.
     */
    public EmbeddingClassifier(MiniLMEmbedder embeddingModel, Map<L, List<String>> examplesByLabel) {
        this(embeddingModel, examplesByLabel, 1, 0.0, 0.5);
    }

    /**
     * Constructs an EmbeddingClassifier with configurable parameters.
     *
     * @param embeddingModel      The MiniLMEmbedder used to generate embeddings.
     * @param examplesByLabel     A map of labels to lists of example texts.
     * @param maxResults          The maximum number of classification results to return.
     * @param minScore            The minimum score threshold for including a classification result.
     * @param meanToMaxScoreRatio The ratio used to weight the mean and max similarity scores for aggregation.
     */
    public EmbeddingClassifier(MiniLMEmbedder embeddingModel, Map<L, List<String>> examplesByLabel, int maxResults, double minScore, double meanToMaxScoreRatio) {
        this.embeddingModel = Objects.requireNonNull(embeddingModel, "embeddingModel cannot be null");
        this.exampleEmbeddingsByLabel = new HashMap<>();
        examplesByLabel.forEach((label, examples) -> {
            List<double[]> embeddings = new ArrayList<>();
            for (String example : examples) {
                embeddings.add(embeddingModel.embed(example));  // Generate embeddings for each example
            }
            exampleEmbeddingsByLabel.put(label, embeddings);
        });

        // Validate parameters
        this.maxResults = ValidationUtils.ensureGreaterThanZero(maxResults, "maxResults");
        this.minScore = ValidationUtils.ensureBetween(minScore, 0.0, 1.0, "minScore");
        this.meanToMaxScoreRatio = ValidationUtils.ensureBetween(meanToMaxScoreRatio, 0.0, 1.0, "meanToMaxScoreRatio");
    }

    /**
     * Classifies an input text by comparing its embedding to the embeddings of labeled examples.
     *
     * @param text The input text to classify.
     * @return A list of labels, sorted by relevance, based on cosine similarity scores.
     */
    @Override
    public List<L> classify(String text) {
        double[] textEmbedding = embeddingModel.embed(text);  // Generate embedding for input text
        List<ScoredLabel<L>> scoredLabels = new ArrayList<>();

        // Calculate scores for each label
        exampleEmbeddingsByLabel.forEach((label, exampleEmbeddings) -> {
            double meanScore = 0.0;
            double maxScore = 0.0;

            for (double[] exampleEmbedding : exampleEmbeddings) {
                double cosineSimilarity = cosineSimilarity(textEmbedding, exampleEmbedding);
                double score = relevanceScore(cosineSimilarity);
                meanScore += score;
                maxScore = Math.max(score, maxScore);
            }

            meanScore /= exampleEmbeddings.size(); // Average score for this label
            double aggregateScore = aggregatedScore(meanScore, maxScore);

            if (aggregateScore >= minScore) {
                scoredLabels.add(new ScoredLabel<>(label, aggregateScore));
            }
        });

        // Sort results by score and limit to maxResults
        scoredLabels.sort(Comparator.comparingDouble(scoredLabel -> 1.0 - scoredLabel.score()));
        List<L> result = new ArrayList<>();
        scoredLabels.stream()
                .limit(maxResults)
                .forEach(scoredLabel -> result.add(scoredLabel.label()));

        return result;
    }

    /**
     * Calculates the cosine similarity between two vectors.
     *
     * @param vec1 The first vector.
     * @param vec2 The second vector.
     * @return The cosine similarity, a value between -1 and 1.
     */
    private double cosineSimilarity(double[] vec1, double[] vec2) {
        double dotProduct = 0.0;
        double magnitudeVec1 = 0.0;
        double magnitudeVec2 = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            magnitudeVec1 += vec1[i] * vec1[i];
            magnitudeVec2 += vec2[i] * vec2[i];
        }
        return dotProduct / (Math.sqrt(magnitudeVec1) * Math.sqrt(magnitudeVec2));
    }

    /**
     * Converts a cosine similarity score (range [-1, 1]) to a relevance score (range [0, 1]).
     *
     * @param cosineSimilarity The cosine similarity score.
     * @return The corresponding relevance score.
     */
    private double relevanceScore(double cosineSimilarity) {
        return (cosineSimilarity + 1.0) / 2.0; // Convert range to [0, 1]
    }

    /**
     * Aggregates the mean and max similarity scores into a single score.
     *
     * @param meanScore The mean similarity score.
     * @param maxScore  The maximum similarity score.
     * @return The aggregated score, weighted by meanToMaxScoreRatio.
     */
    private double aggregatedScore(double meanScore, double maxScore) {
        return meanToMaxScoreRatio * meanScore + (1.0 - meanToMaxScoreRatio) * maxScore;
    }
}