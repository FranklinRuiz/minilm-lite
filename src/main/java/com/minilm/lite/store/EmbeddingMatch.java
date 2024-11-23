package com.minilm.lite.store;

public class EmbeddingMatch<T> {
    private final T item;
    private final double score;

    public EmbeddingMatch(T item, double score) {
        this.item = item;
        this.score = score;
    }

    public T getItem() {
        return item;
    }

    public double getScore() {
        return score;
    }
}
