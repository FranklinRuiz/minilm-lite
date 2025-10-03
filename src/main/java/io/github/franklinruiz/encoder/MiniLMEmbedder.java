package io.github.franklinruiz.encoder;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Arrays;

/**
 * MiniLMEmbedder is a utility class for generating embeddings using the all-MiniLM-L6-v2 model.
 * This class integrates with an ONNX-based encoder to process text and generate high-dimensional embeddings.
 */
public class MiniLMEmbedder {

    // Default paths to the model and tokenizer files located in the resources directory.
    private static final String DEFAULT_MODEL_PATH = "all-minilm-l6-v2.onnx";
    private static final String DEFAULT_TOKENIZER_PATH = "all-minilm-l6-v2-tokenizer.json";

    // Simple LRU cache capacity for memoizing embeddings by normalized text.
    private static final int CACHE_CAPACITY = 512;

    // Instance of the ONNX-based encoder used for generating embeddings.
    private final OnnxBertEncoder encoder;

    // Cache map: normalized text -> embedding (double[])
    private final Map<String, double[]> cache;

    /**
     * Constructs a MiniLMEmbedder using the specified model and tokenizer input streams.
     *
     * @param modelStream     InputStream pointing to the ONNX model file.
     * @param tokenizerStream InputStream pointing to the tokenizer configuration file.
     */
    private MiniLMEmbedder(InputStream modelStream, InputStream tokenizerStream) {
        this.encoder = new OnnxBertEncoder(modelStream, tokenizerStream, OnnxBertEncoder.PoolingMode.MEAN);
        this.cache = new LinkedHashMap<String, double[]>(CACHE_CAPACITY, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, double[]> eldest) {
                return size() > CACHE_CAPACITY;
            }
        };
    }

    /**
     * Creates a default instance of the MiniLMEmbedder using model and tokenizer files located in the resources directory.
     *
     * @return A new instance of MiniLMEmbedder initialized with the default model and tokenizer.
     */
    public static MiniLMEmbedder getDefaultModel() {
        ClassLoader classLoader = MiniLMEmbedder.class.getClassLoader();

        try (
                InputStream modelStream = classLoader.getResourceAsStream(DEFAULT_MODEL_PATH);
                InputStream tokenizerStream = classLoader.getResourceAsStream(DEFAULT_TOKENIZER_PATH)
        ) {
            if (modelStream == null || tokenizerStream == null) {
                throw new IllegalArgumentException("Model or tokenizer files not found in resources!");
            }

            return new MiniLMEmbedder(modelStream, tokenizerStream);
        } catch (IOException e) {
            throw new IllegalArgumentException(e);
        }
    }

    /**
     * Generates an embedding for the given input text.
     * Applies minimal normalization and uses an internal LRU cache for efficiency.
     *
     * @param text The input text to be processed.
     * @return A double array representing the embedding of the input text.
     */
    public synchronized double[] embed(String text) {
        String normalized = normalize(text);
        double[] cached = cache.get(normalized);
        if (cached != null) {
            // Return a copy to avoid external mutation while still benefiting from caching the compute step
            return Arrays.copyOf(cached, cached.length);
        }
        OnnxBertEncoder.EmbeddingAndTokenCount embedding = encoder.embed(normalized);
        double[] result = convertToDoubleArray(embedding.embedding);
        cache.put(normalized, result);
        return Arrays.copyOf(result, result.length);
    }

    // Minimal, language-safe normalization: trim and collapse multiple whitespace into single spaces.
    private String normalize(String input) {
        if (input == null) return "";
        String trimmed = input.trim();
        // Collapse consecutive whitespace to a single space without touching diacritics or casing explicitly
        return trimmed.replaceAll("\\s+", " ");
    }

    /**
     * Converts a float array to a double array.
     *
     * @param array The float array to be converted.
     * @return A double array with the same values as the input float array.
     */
    private double[] convertToDoubleArray(float[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }
}