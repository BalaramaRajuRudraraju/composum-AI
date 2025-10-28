package com.composum.ai.backend.base.service.chat.impl.chatmodel;

import com.google.gson.annotations.SerializedName;

/**
 * Represents the collection of content filter results for different categories.
 */
public class ContentFilterResults {

    /**
     * Content filter result for hate speech.
     */
    @SerializedName("hate")
    private ContentFilterResult hate;

    /**
     * Content filter result for self-harm content.
     */
    @SerializedName("self_harm")
    private ContentFilterResult selfHarm;

    /**
     * Content filter result for sexual content.
     */
    @SerializedName("sexual")
    private ContentFilterResult sexual;

    /**
     * Content filter result for violent content.
     */
    @SerializedName("violence")
    private ContentFilterResult violence;

    // Getters and setters
    public ContentFilterResult getHate() {
        return hate;
    }

    public void setHate(ContentFilterResult hate) {
        this.hate = hate;
    }

    public ContentFilterResult getSelfHarm() {
        return selfHarm;
    }

    public void setSelfHarm(ContentFilterResult selfHarm) {
        this.selfHarm = selfHarm;
    }

    public ContentFilterResult getSexual() {
        return sexual;
    }

    public void setSexual(ContentFilterResult sexual) {
        this.sexual = sexual;
    }

    public ContentFilterResult getViolence() {
        return violence;
    }

    public void setViolence(ContentFilterResult violence) {
        this.violence = violence;
    }
}