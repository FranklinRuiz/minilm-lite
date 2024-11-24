package io.github.franklinruiz.store;

public class TextSegment implements Embeddable {
    private final String content;

    public TextSegment(String content) {
        this.content = content;
    }

    @Override
    public String getText() {
        return content;
    }
}
