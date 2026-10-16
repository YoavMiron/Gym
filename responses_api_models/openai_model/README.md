# Description

OpenAI-compatible model server using Gym's shared HTTP client.

## HTTP retries

HTTP 404 and 408 are retryable by default, alongside the existing transient
server and rate-limit errors. This applies to every dataset and to policy and
judge clients. A permanent 404 still fails after the bounded attempt budget.
Retries resend the same request with the existing fixed 0.5-second delay
between attempts and preserve the terminal error body.

`max_http_attempts` defaults to three total attempts. Set it per model server:

```yaml
judge_model:
  responses_api_models:
    openai_model:
      max_http_attempts: 5
```

A value of one disables HTTP retries. Connection-error retries are controlled
separately. Internal Gym clients retain their existing unbounded extension for
rate-limit statuses; 404 and 408 never trigger that extension.

## Reasoning items from gateways

LiteLLM-style gateways (for example in front of Anthropic and Google models)
return Responses API `reasoning` output items whose `summary` is null or absent
(Gym's own client saw the key missing: `output.0.reasoning.summary Field
required`; the `openai` client's dump of the same bodies shows `summary: null`)
and the reasoning text in `output_text` content parts. The Responses schema, and so
`NeMoGymResponse`, requires `summary` to be a list and those parts to be
`reasoning_text`, so every such response failed validation and its rollout was
lost. `normalize_reasoning_items` (default `true`) repairs exactly those two
shapes before validation and changes nothing else (`encrypted_content` is
kept). Set it to `false` to see the raw provider shape as a validation error:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      normalize_reasoning_items: false
```

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
