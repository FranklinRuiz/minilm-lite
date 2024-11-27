package io.github.franklinruiz.classifier;

import java.util.List;

public interface TextClassifier<L> {
    List<L> classify(String text);
}
