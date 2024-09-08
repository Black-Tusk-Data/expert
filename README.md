# Expert

Expert is the head-end of an information retrieval system.
It can build 'knowledge bases' over a variety of information sources.

Knowledge bases are a collection of information about a specific topic that support:
 - conceptual searching
 - AI domain expert chat

# Usage

## API keys

Expert, at this time, relies on third-party LLM hosting to compute inference or embeddings.

By default, it will expect a `GROQ_API_KEY` and `JINA_AI_API_KEY`.
There is support for a number of other hosted model providers, but this is not exposed via the CLI at least right now.


## Building a knowledge base

```sh
expert build --kb ./linear_algebra.kb --doc ./linear-algebra-done-wrong.pdf
```


## Talking to the knowledge base

```sh
expert query --kb ./linear_algebra.kb --query "Does the cross product only make sense in 3 dimensions?" --verbose
```

# Coming soon

 - CLI support for model selection
 - HTTP server interface
