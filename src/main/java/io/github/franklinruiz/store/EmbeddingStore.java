package io.github.franklinruiz.store;

import io.github.franklinruiz.encoder.MiniLMEmbedder;
import io.github.franklinruiz.utils.CosineSimilarityUtil;

import java.util.*;
import java.util.function.ToDoubleFunction;

/**
 * EmbeddingStore is a storage and retrieval system for items and their embeddings.
 * It allows you to add items, compute their embeddings using a MiniLMEmbedder, and find relevant items
 * based on a query embedding using cosine similarity.
 *
 * @param <T> The type of items stored, which must implement the {@link Embeddable} interface to provide text representation.
 */
public class EmbeddingStore<T extends Embeddable> {

    // List of items stored in the EmbeddingStore.
    private final List<T> items;

    // List of embeddings corresponding to the stored items.
    private final List<double[]> embeddings;

    // MiniLMEmbedder used to compute embeddings for the items.
    private final MiniLMEmbedder embedder;

    /**
     * Constructs an EmbeddingStore with the specified MiniLMEmbedder.
     *
     * @param embedder An instance of MiniLMEmbedder used to compute embeddings.
     */
    private EmbeddingStore(MiniLMEmbedder embedder) {
        this.embedder = embedder;
        this.items = new ArrayList<>();
        this.embeddings = new ArrayList<>();
    }

    /**
     * Initializes an EmbeddingStore with a default MiniLMEmbedder.
     *
     * @param <T> The type of items to store, which must implement {@link Embeddable}.
     * @return A new instance of EmbeddingStore initialized with a default MiniLMEmbedder.
     */
    public static <T extends Embeddable> EmbeddingStore<T> initialize() {
        try {
            MiniLMEmbedder embedder = MiniLMEmbedder.getDefaultModel();
            return new EmbeddingStore<>(embedder);
        } catch (Exception e) {
            throw new IllegalArgumentException("Error initializing EmbeddingStore", e);
        }
    }

    /**
     * Adds an item to the EmbeddingStore. The item's embedding is computed and stored.
     *
     * @param item The item to add. Must implement the {@link Embeddable} interface to provide text for embedding.
     */
    public void addItem(T item) {
        try {
            double[] embedding = embedder.embed(item.getText());
            items.add(item);
            embeddings.add(embedding);
        } catch (Exception e) {
            throw new IllegalArgumentException("Error adding item to EmbeddingStore", e);
        }
    }

    /**
     * Finds the most relevant items in the store based on their similarity to the query embedding.
     *
     * @param queryEmbedding The query embedding to compare against the stored embeddings.
     * @param maxResults     The maximum number of relevant items to retrieve.
     * @return A list of {@link EmbeddingMatch} objects, sorted by similarity in descending order.
     */
    public List<EmbeddingMatch<T>> findRelevant(double[] queryEmbedding, int maxResults) {
        // Priority queue to maintain the top relevant matches, sorted by score in descending order.
        PriorityQueue<EmbeddingMatch<T>> matches = new PriorityQueue<>(
                Comparator.comparingDouble((ToDoubleFunction<EmbeddingMatch<T>>) EmbeddingMatch::getScore).reversed()
        );

        // Compute similarity for each embedding in the store.
        for (int i = 0; i < embeddings.size(); i++) {
            double similarity = CosineSimilarityUtil.calculate(queryEmbedding, embeddings.get(i));
            matches.offer(new EmbeddingMatch<>(items.get(i), similarity));
        }

        // Collect the top matches.
        List<EmbeddingMatch<T>> results = new ArrayList<>();
        for (int i = 0; i < maxResults && !matches.isEmpty(); i++) {
            results.add(matches.poll());
        }
        return results;
    }
}