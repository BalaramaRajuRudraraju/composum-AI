package com.composum.ai.backend.base.service.chat.impl.chatmodel;

import com.google.gson.annotations.SerializedName;

/**
 * Represents content filter results for a specific content category (hate, self_harm, sexual, violence).
 */
public class ContentFilterResult {

    /**
     * Indicates whether the content was filtered.
     */
    @SerializedName("filtered")
    private boolean filtered;

    /**
     * The severity level of the content (e.g., "safe", "low", "medium", "high").
     */
    @SerializedName("severity")
    private String severity;

    // Getters and setters
    public boolean isFiltered() {
        return filtered;
    }

    public void setFiltered(boolean filtered) {
        this.filtered = filtered;
    }

    public String getSeverity() {
        return severity;
    }

    public void setSeverity(String severity) {
        this.severity = severity;
    }
}