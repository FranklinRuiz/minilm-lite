package io.github.franklinruiz.encoder;

import java.io.IOException;
import java.io.InputStream;

/**
 * MiniLMEmbedder is a utility class for generating embeddings using the all-MiniLM-L6-v2 model.
 * This class integrates with an ONNX-based encoder to process text and generate high-dimensional embeddings.
 */
public class MiniLMEmbedder {

    // Default paths to the model and tokenizer files located in the resources directory.
    private static final String DEFAULT_MODEL_PATH = "all-minilm-l6-v2.onnx";
    private static final String DEFAULT_TOKENIZER_PATH = "all-minilm-l6-v2-tokenizer.json";

    // Instance of the ONNX-based encoder used for generating embeddings.
    private final OnnxBertEncoder encoder;

    /**
     * Constructs a MiniLMEmbedder using the specified model and tokenizer input streams.
     *
     * @param modelStream     InputStream pointing to the ONNX model file.
     * @param tokenizerStream InputStream pointing to the tokenizer configuration file.
     */
    private MiniLMEmbedder(InputStream modelStream, InputStream tokenizerStream) {
        this.encoder = new OnnxBertEncoder(modelStream, tokenizerStream, OnnxBertEncoder.PoolingMode.MEAN);
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
     *
     * @param text The input text to be processed.
     * @return A double array representing the embedding of the input text.
     */
    public double[] embed(String text) {
        // Generates embeddings using the ONNX encoder and converts them to a double array.
        OnnxBertEncoder.EmbeddingAndTokenCount embedding = encoder.embed(text);
        return convertToDoubleArray(embedding.embedding);
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