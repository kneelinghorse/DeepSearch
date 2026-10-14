# Quality Reference: Research Output Examples

This file documents the quality standards for DeepSearch research output
after the B6.3 prompt engineering overhaul.

## BAD OUTPUT (Chain-of-Thought Leaking)

This is what the output looked like BEFORE the fix:

```markdown
## Research Findings

### Official Documentation Review
No findings captured for this phase.

### Additional Insights
1. Okay, I understand the mission. I need to investigate the Gemini API structure.
2. First, I'll use the web_search tool to look for documentation about authentication.
3. Let me search for the rate limits and pricing information...
4. Based on my understanding of the mission, I should focus on the grounding feature.
5. I will start by examining the generateContent endpoint parameters.
```

**Problems:**
- Lines 1-5 are internal reasoning, not findings
- No actual information about the API
- No citations or sources
- Reads like a process log, not research

## GOOD OUTPUT (Synthesized Findings)

This is what the output should look like AFTER the fix:

```markdown
## Research Findings

### API Authentication
1. The Gemini API uses API keys as the primary authentication method for server-to-server calls, with OAuth 2.0 available for user-context requests [Google AI Documentation](https://ai.google.dev/docs/auth).

### Rate Limits and Pricing
2. Free tier rate limits are set at 60 requests per minute (RPM) with 1 million tokens per month, while paid tiers allow up to 1000 RPM [Pricing Guide](https://ai.google.dev/pricing).

### Grounding Capabilities
3. Gemini 2.0 models support grounded search via the `tools.google_search` configuration, replacing the deprecated `googleSearchRetrieval` tool [API Reference](https://ai.google.dev/api).
4. The `generateContent` endpoint accepts a `tools` array for enabling search grounding, with results including `groundingMetadata` when citations are available.

### Streaming Support
5. Both synchronous and streaming modes are supported via the `stream` parameter or the `:streamGenerateContent` endpoint variant [Streaming Guide](https://ai.google.dev/docs/streaming).
```

**Improvements:**
- Each finding is a factual statement
- Inline citations link claims to sources
- Organized by topic/theme
- Professional research quality
- No internal reasoning exposed

## Validation Criteria

A research output PASSES quality check if:
- [ ] No findings start with "Okay,", "First,", "Let me", "I will", etc.
- [ ] Each finding contains substantive information
- [ ] Citations are present (>50% of findings should have sources)
- [ ] Findings are organized logically
- [ ] No tool usage narration ("Using web_search to...")

## Defense-in-Depth

Even with improved prompts, the system includes a filter that removes
thinking patterns. The following patterns are automatically filtered:

- "okay, i" / "ok, i" / "alright, i"
- "let me" / "i will" / "i'll"
- "first, i" / "now i"
- "i need to" / "i should" / "i understand"
- "based on the mission" / "based on my understanding"
- "using web_search" / "searching for"
- "my plan is" / "my approach"

This ensures clean output even if the LLM occasionally reverts to
chain-of-thought patterns.
