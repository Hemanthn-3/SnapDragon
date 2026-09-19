# NEXUS Evidence Verification Layer (Phase 7)

**Offline Multimodal AI Work Agent for Snapdragon PCs**  
**Specification & Verification Architecture**

---

## 1. Overview & Verification Philosophy

In high-stakes enterprise, industrial, and technical workflows on Snapdragon-powered Windows PCs, **unsubstantiated claims and ungrounded LLM hallucinations are unacceptable**.

The **Evidence Verification Layer** ensures that:
1. **Every important generated finding must be linked to verifiable local evidence.**
2. **Missing evidence $\rightarrow$ NEVER `VERIFIED`**: If no evidence chunks exist or are retrieved, the claim is classified as `INSUFFICIENT_EVIDENCE`.
3. **Contradictions $\rightarrow$ NEVER `VERIFIED`**: If retrieved documents contradict the claim (e.g. opposite polarity, numbers within threshold vs exceeding threshold), the finding is classified as `UNCERTAIN` or `INSUFFICIENT_EVIDENCE`.
4. **Multi-Source Corroboration**: To attain the highest trust level (`VERIFIED`), findings must be corroborated across multiple distinct documents or pages.
5. **Concise Evidence Explanation (Zero Chain-of-Thought Leaks)**: The agent must never expose internal chain-of-thought tokens (e.g. `<think>`, scratchpad tokens, prompt templates). Instead, it outputs clean, concise citations.

---

## 2. Core Data Models

### 2.1 `EvidenceItem`
Represents an immutable, traceable reference to an extracted chunk within the local document repository:

```json
{
  "document": "Safety_Manual.pdf",
  "page": 17,
  "source_chunk": "chunk_018",
  "relevant_text": "Maximum allowable bearing vibration threshold is 8.0 mm/s RMS."
}
```

- **`document`**: Ingested document filename.
- **`page`**: Page number where evidence chunk was extracted.
- **`source_chunk`**: Unique chunk identifier in SQLite database.
- **`relevant_text`**: Exact textual excerpt supporting or refuting the claim.

### 2.2 `Finding`
The primary structured unit of verified insight:

```json
{
  "id": "finding_1",
  "claim": "Inspection value exceeds the reference threshold.",
  "evidence": [
    {
      "document": "Safety_Manual.pdf",
      "page": 17,
      "source_chunk": "chunk_018",
      "relevant_text": "Maximum allowable bearing vibration threshold is 8.0 mm/s RMS."
    },
    {
      "document": "Inspection_Report.pdf",
      "page": 4,
      "source_chunk": "chunk_042",
      "relevant_text": "Turbine bearing 2 measured vibration: 14.2 mm/s RMS."
    }
  ],
  "confidence": 0.94,
  "verification_status": "VERIFIED",
  "explanation": "Evidence:\nInspection_Report.pdf — Page 4\nSafety_Manual.pdf — Page 17"
}
```

---

## 3. Verification Status Taxonomy & Transition Rules

```
                      [Claim Under Evaluation]
                                 |
                 +---------------+---------------+
                 |                               |
        [No Evidence Found]             [Evidence Found]
                 |                               |
                 v                               v
      INSUFFICIENT_EVIDENCE             [Check Contradiction]
      (Confidence: 0.0 - 0.2)                    |
                                 +---------------+---------------+
                                 |                               |
                          [Contradiction]               [No Contradiction]
                                 |                               |
                                 v                               v
                             UNCERTAIN               [Compute Semantic Match]
                       (Confidence: 0.1 - 0.25)                  |
                                                 +---------------+---------------+
                                                 |                               |
                                           [Single Source]               [Multi-Source]
                                                 |                               |
                                                 v                               v
                                             SUPPORTED                       VERIFIED
                                       (Confidence: 0.7 - 0.85)        (Confidence: 0.88 - 0.96)
```

| Status | Description | Confidence Range | Minimum Evidence Requirement |
| :--- | :--- | :---: | :--- |
| **`VERIFIED`** | Claim is corroborated across multiple distinct documents or pages with high semantic and factual alignment. | `0.88 – 0.96` | $\ge 2$ distinct documents/pages with matching evidence |
| **`SUPPORTED`** | Claim is grounded by a single reliable, non-contradicted source chunk. | `0.70 – 0.85` | 1 reliable matching document chunk |
| **`UNCERTAIN`** | Retrieved evidence conflicts with or contradicts the claim, or semantic similarity is ambiguous. | `0.10 – 0.25` | Contradiction detected or ambiguous polarity |
| **`INSUFFICIENT_EVIDENCE`**| No evidence exists in local storage, or retrieved chunks fail to substantiate the claim. | `0.00 – 0.20` | 0 matching chunks or missing citations |

---

## 4. Contradiction Detection Rules

A claim is **strictly barred from `VERIFIED` or `SUPPORTED`** if any of the following are true:

1. **Polarity Conflict**:
   - Claim asserts non-compliance or failure (*"exceeds"*, *"violated"*, *"unsafe"*, *"failed"*), while evidence states compliance (*"within threshold"*, *"compliant"*, *"safe"*, *"passed inspection"*, *"no deviation"*).
   - Claim asserts compliance, while evidence states an excursion or failure.
2. **Numerical & Threshold Contradiction**:
   - Measured value reported in evidence does not exceed the limit cited, yet the claim states it did.
3. **Explicit Negation Mismatch**:
   - Syntactic negation polarity inverted between claim and source excerpt.

When a contradiction is detected, the finding is assigned `UNCERTAIN` with an explanation documenting the exact conflict.

---

## 5. Explainability & Privacy Policy

> [!CAUTION]
> **Zero Exposure of Hidden Chain-of-Thought**
> Under Qualcomm / Snapdragon privacy and user experience standards:
> - Raw LLM prompt wrappers, internal reasoning traces (`<think>...</think>`), and scratchpads **must never be displayed to the user**.
> - The `explanation` field must contain concise citations of verified sources formatted as:
>   ```
>   Evidence:
>   Safety_Manual.pdf — Page 17
>   Inspection_Report.pdf — Page 4
>   ```
> - In case of contradiction or insufficient evidence, output a clear, factual reason:
>   ```
>   Contradiction identified: Inspection_Report.pdf (Page 4) reports conflicting data (Evidence states 'within threshold').
>   ```

---

## 6. Offline Implementation & Snapdragon Acceleration

The Evidence Verification Layer runs **100% offline**:
- **Vector Embeddings**: `all-MiniLM-L6-v2` ONNX model quantized to INT8, optimized for Qualcomm Hexagon NPU / CPU execution.
- **Storage**: SQLite database in WAL mode (`nexus.db`), querying dense chunk embeddings via local vector dot-products.
- **REST Endpoints**:
  - `POST /verification/verify`: Single claim verification.
  - `POST /verification/batch`: Batch multi-finding verification.
