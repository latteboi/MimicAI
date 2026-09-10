"""Provider adapters, split by wire format.

`api_service.py` next door keeps `APIService` -- routing, fallback, pricing and the
key-cooldown bookkeeping. Everything here is the shape of one provider's HTTP:

    rest_view.py   attribute views over parsed REST JSON, shared by all three
    streaming.py   request bodies that carry files without buffering them
    openrouter.py  the OpenRouter adapter
    ollama.py      the Ollama adapter
    google_rest.py the Google adapter, plus inline-blob and TTS handling
    embeddings.py  query embeddings and their cache

The dependency direction is one way: rest_view and streaming know nothing about
providers, and no adapter imports another.
"""
