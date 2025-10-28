package com.composum.ai.backend.base.service.chat.impl.chatmodel;

import com.google.gson.annotations.SerializedName;

/**
 * Represents prompt filter results for a specific prompt index.
 */
public class PromptFilterResult {

    /**
     * The index of the prompt that was filtered.
     */
    @SerializedName("prompt_index")
    private int promptIndex;

    /**
     * The content filter results for this prompt.
     */
    @SerializedName("content_filter_results")
    private ContentFilterResults contentFilterResults;

    // Getters and setters
    public int getPromptIndex() {
        return promptIndex;
    }

    public void setPromptIndex(int promptIndex) {
        this.promptIndex = promptIndex;
    }

    public ContentFilterResults getContentFilterResults() {
        return contentFilterResults;
    }

    public void setContentFilterResults(ContentFilterResults contentFilterResults) {
        this.contentFilterResults = contentFilterResults;
    }
}