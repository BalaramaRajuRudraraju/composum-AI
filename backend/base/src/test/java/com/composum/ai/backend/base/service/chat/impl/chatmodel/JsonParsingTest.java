package com.composum.ai.backend.base.service.chat.impl.chatmodel;

import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.ErrorCollector;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import static org.hamcrest.CoreMatchers.*;

/**
 * Test class to verify JSON parsing of content filter responses
 */
public class JsonParsingTest {
    
    @Rule
    public ErrorCollector ec = new ErrorCollector();
    
    @Test
    public void testContentFilterJsonParsing() {
        Gson gson = new GsonBuilder().disableHtmlEscaping().create();
        
        String testJson = "{\"choices\":[],\"created\":0,\"id\":\"\",\"model\":\"\",\"object\":\"\",\"prompt_filter_results\":[{\"prompt_index\":0,\"content_filter_results\":{\"hate\":{\"filtered\":false,\"severity\":\"safe\"},\"self_harm\":{\"filtered\":false,\"severity\":\"safe\"},\"sexual\":{\"filtered\":false,\"severity\":\"safe\"},\"violence\":{\"filtered\":false,\"severity\":\"safe\"}}}]}";
        
        try {
            ChatCompletionResponse response = gson.fromJson(testJson, ChatCompletionResponse.class);
            ec.checkThat("Response should not be null", response, notNullValue());
            ec.checkThat("ID should be empty string", response.getId(), is(""));
            ec.checkThat("Object should be empty string", response.getObject(), is(""));
            ec.checkThat("Model should be empty string", response.getModel(), is(""));
            ec.checkThat("Created should be 0", response.getCreated(), is(0L));
            ec.checkThat("Choices should be empty list", response.getChoices(), notNullValue());
            ec.checkThat("Choices should be empty", response.getChoices().size(), is(0));
            ec.checkThat("Prompt filter results should not be null", response.getPromptFilterResults(), notNullValue());
            ec.checkThat("Should have one prompt filter result", response.getPromptFilterResults().size(), is(1));
            
            PromptFilterResult filterResult = response.getPromptFilterResults().get(0);
            ec.checkThat("Prompt index should be 0", filterResult.getPromptIndex(), is(0));
            ec.checkThat("Content filter results should not be null", filterResult.getContentFilterResults(), notNullValue());
            
            ContentFilterResults contentFilters = filterResult.getContentFilterResults();
            ec.checkThat("Hate filter should not be null", contentFilters.getHate(), notNullValue());
            ec.checkThat("Hate should not be filtered", contentFilters.getHate().isFiltered(), is(false));
            ec.checkThat("Hate severity should be safe", contentFilters.getHate().getSeverity(), is("safe"));
            
        } catch (Exception e) {
            ec.addError(new AssertionError("JSON parsing failed: " + e.getMessage(), e));
        }
    }
}