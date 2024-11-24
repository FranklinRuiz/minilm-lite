package io.github.franklinruiz.encoder;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.LongBuffer;
import java.util.*;

/**
 * OnnxBertEncoder is a class that processes text to generate embeddings using a pre-trained ONNX-based BERT model.
 * It supports tokenization, embedding generation, and pooling strategies for text processing.
 */
public class OnnxBertEncoder {

    // Maximum sequence length allowed for input tokens.
    private static final int MAX_SEQUENCE_LENGTH = 510;

    // ONNX Runtime environment for managing the model session.
    private final OrtEnvironment environment;

    // ONNX Runtime session for executing the model.
    private final OrtSession session;

    // Set of expected input names for the ONNX model.
    private final Set<String> expectedInputs;

    // Tokenizer for text preprocessing, compatible with the Hugging Face format.
    private final HuggingFaceTokenizer tokenizer;

    // Pooling mode to determine how embeddings are aggregated.
    private final PoolingMode poolingMode;

    /**
     * Constructs an OnnxBertEncoder with the specified model, tokenizer, and pooling mode.
     *
     * @param model       InputStream representing the ONNX model file.
     * @param tokenizer   InputStream representing the tokenizer configuration file.
     * @param poolingMode PoolingMode to determine the aggregation strategy (e.g., CLS or MEAN).
     */
    public OnnxBertEncoder(InputStream model, InputStream tokenizer, PoolingMode poolingMode) {
        try {
            this.environment = OrtEnvironment.getEnvironment();
            this.session = this.environment.createSession(this.loadModel(model));
            this.expectedInputs = this.session.getInputNames();
            this.tokenizer = HuggingFaceTokenizer.newInstance(tokenizer, Collections.singletonMap("padding", "false"));
            this.poolingMode = Objects.requireNonNull(poolingMode, "Pooling mode cannot be null");
        } catch (Exception e) {
            throw new IllegalArgumentException(e);
        }
    }

    /**
     * Generates an embedding for the given input text.
     *
     * @param text The input text to process.
     * @return An EmbeddingAndTokenCount object containing the embedding vector and token count.
     */
    public EmbeddingAndTokenCount embed(String text) {
        List<String> tokens = this.tokenizer.tokenize(text);
        List<List<String>> partitions = partition(tokens, MAX_SEQUENCE_LENGTH);
        List<float[]> embeddings = new ArrayList<>();

        for (List<String> partition : partitions) {
            try (OrtSession.Result result = this.encode(partition)) {
                float[] embedding = this.toEmbedding(result);
                embeddings.add(embedding);
            } catch (OrtException e) {
                throw new IllegalArgumentException(e);
            }
        }

        List<Integer> weights = partitions.stream().map(List::size).toList();
        float[] embedding = normalize(this.weightedAverage(embeddings, weights));
        return new EmbeddingAndTokenCount(embedding, tokens.size());
    }

    /**
     * Counts the number of tokens in the given text after tokenization.
     *
     * @param text The input text to tokenize.
     * @return The number of tokens in the input text.
     */
    public int countTokens(String text) {
        return this.tokenizer.tokenize(text).size();
    }

    // Encodes the tokenized input into ONNX model tensors and runs inference.
    private OrtSession.Result encode(List<String> tokens) throws OrtException {
        Encoding encoding = this.tokenizer.encode(this.toText(tokens), true, false);
        long[] inputIds = encoding.getIds();
        long[] attentionMask = encoding.getAttentionMask();
        long[] tokenTypeIds = encoding.getTypeIds();
        long[] shape = new long[]{1L, inputIds.length};

        try (
                OnnxTensor inputIdsTensor = OnnxTensor.createTensor(this.environment, LongBuffer.wrap(inputIds), shape);
                OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(this.environment, LongBuffer.wrap(attentionMask), shape);
                OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(this.environment, LongBuffer.wrap(tokenTypeIds), shape)
        ) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);
            if (this.expectedInputs.contains("token_type_ids")) {
                inputs.put("token_type_ids", tokenTypeIdsTensor);
            }
            return this.session.run(inputs);
        }
    }

    // Converts tokens into text while removing special tokens if necessary.
    private String toText(List<String> tokens) {
        String text = this.tokenizer.buildSentence(tokens);
        List<String> tokenized = this.tokenizer.tokenize(text);
        List<String> tokenizedWithoutSpecialTokens = new LinkedList<>(tokenized);
        tokenizedWithoutSpecialTokens.remove(0);
        tokenizedWithoutSpecialTokens.remove(tokenizedWithoutSpecialTokens.size() - 1);
        return tokenizedWithoutSpecialTokens.equals(tokens) ? text : String.join("", tokens);
    }

    // Converts the ONNX result into a pooled embedding vector.
    private float[] toEmbedding(OrtSession.Result result) throws OrtException {
        float[][] vectors = ((float[][][]) result.get(0).getValue())[0];
        return this.pool(vectors);
    }

    // Applies the specified pooling mode to the embedding vectors.
    private float[] pool(float[][] vectors) {
        return switch (this.poolingMode) {
            case CLS -> clsPool(vectors);
            case MEAN -> meanPool(vectors);
        };
    }

    // Performs CLS pooling on the embedding vectors.
    private static float[] clsPool(float[][] vectors) {
        return vectors[0];
    }

    // Performs mean pooling on the embedding vectors.
    private static float[] meanPool(float[][] vectors) {
        int numVectors = vectors.length;
        int vectorLength = vectors[0].length;
        float[] averagedVector = new float[vectorLength];

        for (float[] vector : vectors) {
            for (int j = 0; j < vectorLength; ++j) {
                averagedVector[j] += vector[j];
            }
        }

        for (int j = 0; j < vectorLength; ++j) {
            averagedVector[j] /= numVectors;
        }

        return averagedVector;
    }

    // Computes the weighted average of embeddings based on token weights.
    private float[] weightedAverage(List<float[]> embeddings, List<Integer> weights) {
        int dimensions = embeddings.get(0).length;
        float[] averagedEmbedding = new float[dimensions];
        int totalWeight = weights.stream().mapToInt(Integer::intValue).sum();

        for (int i = 0; i < embeddings.size(); ++i) {
            int weight = weights.get(i);

            for (int j = 0; j < dimensions; ++j) {
                averagedEmbedding[j] += embeddings.get(i)[j] * weight;
            }
        }

        for (int j = 0; j < dimensions; ++j) {
            averagedEmbedding[j] /= totalWeight;
        }

        return averagedEmbedding;
    }

    // Normalizes the embedding vector to have unit norm.
    private static float[] normalize(float[] vector) {
        float sumSquare = 0.0F;

        for (float v : vector) {
            sumSquare += v * v;
        }

        float norm = (float) Math.sqrt(sumSquare);
        float[] normalizedVector = new float[vector.length];

        for (int i = 0; i < vector.length; ++i) {
            normalizedVector[i] = vector[i] / norm;
        }

        return normalizedVector;
    }

    // Loads the model from the InputStream and converts it into a byte array.
    private byte[] loadModel(InputStream modelInputStream) {
        try (ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {
            byte[] data = new byte[1024];

            int nRead;
            while ((nRead = modelInputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }

            buffer.flush();
            return buffer.toByteArray();
        } catch (IOException e) {
            throw new IllegalArgumentException(e);
        }
    }

    // Partitions tokens into segments that fit within the model's input size.
    private static List<List<String>> partition(List<String> tokens, int partitionSize) {
        List<List<String>> partitions = new ArrayList<>();

        int to;
        for (int from = 1; from < tokens.size() - 1; from = to) {
            to = from + partitionSize;
            if (to >= tokens.size() - 1) {
                to = tokens.size() - 1;
            } else {
                while (tokens.get(to).startsWith("##")) {
                    --to;
                }
            }
            partitions.add(tokens.subList(from, to));
        }
        return partitions;
    }

    /**
     * Enum to define the pooling mode for embeddings.
     */
    public enum PoolingMode {
        CLS, MEAN
    }

    /**
     * A helper class to encapsulate an embedding and the token count.
     */
    public static class EmbeddingAndTokenCount {
        public final float[] embedding;
        public final int tokenCount;

        public EmbeddingAndTokenCount(float[] embedding, int tokenCount) {
            this.embedding = embedding;
            this.tokenCount = tokenCount;
        }
    }
}