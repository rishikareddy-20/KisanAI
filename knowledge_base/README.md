# Knowledge Base

`diseases.json` is the curated agronomy knowledge base that grounds every piece of
advice KisanAI gives. The treatment content is authored and curated here. The
advisory layer reads it directly rather than generating treatments.

## Structure

```jsonc
{
  "languages": ["en", "hi"],
  "labels": {
    "section":   { "en": {...}, "hi": {...}},                // section headings
    "uncertain": { "en": "...", ... },                       // low-confidence message
    "healthy":   { "en": "...", ... },                       // healthy-leaf message
    "safety":    { "en": "...", ... }                        // pesticide safety note
  },
  "classes": {
    "Tomato___Late_blight": {
      "crop": "Tomato",
      "healthy": false,
      "translations": {
        "en": {
          "name": "Tomato Late Blight",
          "summary": "...",
          "symptoms":   ["..."],
          "organic":    ["..."],
          "chemical":   ["..."],
          "prevention": ["..."],
          "severity": "high"
        },
        "hi": { ... }   // Hindi content for every non-healthy class
      }
    }
  }
}
```

## How it's used
- Class keys (for example `Tomato___Late_blight`) match the vision model's output
  labels exactly, so retrieval is a direct 1:1 lookup.
- The template provider renders these fields into a structured plan.
- The optional semantic index (`backend/rag/build_index.py`) embeds these entries so
  farmers can also search by free-text symptoms.

## Extending
1. Add a new class object under `"classes"` with the exact vision label as the key.
2. Provide both the `en` and `hi` translations. English is the fallback, but every
   non-healthy class should carry Hindi so Hindi advice is never served in English.
3. Rebuild the semantic index if you use it: `python -m backend.rag.build_index`.

> Note: the treatment content here is a starter set for development. Before any real
> deployment, have the dosages, chemicals, and practices reviewed by a qualified
> agronomist and localized to the target region's approved inputs.
