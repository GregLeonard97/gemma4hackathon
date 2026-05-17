Introduction
Postgraduate medical training in the United Kingdom involves frequent rotations throughout different hospitals and National Health Service (NHS) trusts. These moves across hospitals can be as frequent as monthly at the most demanding times and can commonly occur six monthly. Although underlying pathophysiology and management does not change, local guidelines for choice of medication, dosing regimens and specific follow-up pathways vary between trust. 

Information about specific local guidelines are stored haphazardly across local intranet systems. Slow workstation hardware compounds the friction. Hilton (1) suggests that if each doctor in the NHS spends just 10 minutes a day waiting of NHS systems, the annual cost of this may exceed £140,000,000 per annum. 

Using smaller Gemma 4 (E2B, E4B) models, I built a clinically-grounded RAG augmented app to answer process-driven clinical queries in natural language – citing guidelines from source PDFs to ensure clinicians are confident answers are generated from ground truths. 

Design
An entirely local setup was necessary to ensure no violation of NHS data governance and reduce risk. Built in Swift, the app runs entirely on device after a one-time model download (tested: iPhone 17 Pro Max). The app answers clinical questions in natural language, returns answers sourced verbatim from neonatal guidelines and streams token-by-token. Each answer cites text from a certain guideline, tapping on said guideline will open the PDF on the page cited allowing further interrogation of the guideline.

164-PDF corpus of guidelines (drawn from the UCLH neonatal unit’s clinical guideline library where I work) is processed on mac into a compressed, on-device vector database of 10,022 chunks occupying 9.69MB (SQLite) plus 60MB of rendered table and flowchart images. Clinical queries embed locally via BGE-small-en-v1.5, retrieve top-5 chunks via cosine similarity over TurboQuant-3-bit-compressed vectors and feed Gemma 4 E2B running on the MLX framework. 

The model generates source-cited answers from retrieved excerpts under a 12-rule clinical safety system prompt, with refusal by default when retrieved evidence is insufficient. 

Gemma 4 E2B at 4-bit quantisation operates within iPhone 17 Pro Max’s per-process memory budget. Responses are generated consistently within 2-5 seconds. E4B was explored as an option although exceeded memory budget consistently (by approximately 200MB even after aggressive memory optimisation) with clinically relevant prompts so was unable to be implemented in the working model.
Technical Execution

On Mac, a document ingestion pipeline takes the corpus of 164 UCLH specific PDFs into searchable chunks once. PyMuPDF is used for text extraction (selected specifically for table layout fidelity) with per-page concatenation before splitting to preserve context. LangChain RecursiveCharacterTextSplitter is used to produce 800-character chunks with 150-character overlap. Larger chunks improved retrieval context while preserving R@5 precision.

Given the tendency of clinical guidelines to store information within tables they were prioritised. Each table is detected during ingestion producing three artefacts; row-level chunks (one per row with column headers preserved), a table summary chunk and a rendered PNG for retrieval of the full table. Consequently, precise queries such as “drug dose of x-gestation neonate” hits specific table rows, “show me parenteral nutrition table for x-gestation” generates the rendered image. 

A key feature is the pipeline is hospital-agnostic by design. Replacing the bundled PDF corpus with another trust's guidelines and rebuilding the vector database produces a working tool for that site or speciality with minimal prompt adjustment.

A single embedding model was used for indexing and querying. I used BAAI/bge-small-en-v1.5. The clinical corpus is small (164 PDFs), and bge-small (384-dim, ~130MB Core ML package) is sufficient for semantic retrieval at this scale while fitting in the on-device memory budget. TurboQuant (2) was implemented at 3-bit precision. Testing R@5 across 1-, 2-, 3-, 4-bit, and 5-bit quantisation, plus uncompressed baseline showed 3-bit quantisation achieved the highest R@5 (0.9787), better than the uncompressed baseline (0.89) when applied to the eval dataset. This allowed near 10x compression of vectors on device (60MB -> 7.3MB + vector indices). 

A level of query normalisation was implemented to capture clinical shorthand using bidirectional abbreviation at retrieval time (e.g. “EOS” and “early onset sepsis” both query). Keyword heuristic detection combined with retrieval metadata improves recall of specific algorithms/tables e.g. “show me the amikacin dosing table” returns a one-sentence summary referencing the table location, and the UI surfaces the rendered table image as a zoomable view.

Clinical safety constraints were ensured via prompt engineering rather than fine-tuning to ensure the app could be replicated across different departments/hospitals with minor changes, rather than training an entirely new model. The prompt structure covers clinical language interpretation (“follow-up” = post discharge), verbatim rules for doses and threshold with calculation safety, semantic logic regarding the use of dates in neonatology (i.e. in neonatology, '28 weeks' refers to gestational age by convention, not postnatal age) and refusal calibration on inappropriate queries or queries where there is inadequate evidence to respond.

Gemma 4 E2B (mlx-community/gemma-4-e2b-it-4bit) temperature 0.1, maxTokens 1024 is used as the grounded answer generator. E4B was initially targeted but proved unsuitable on-device: queries beyond single-word triggered out-of-memory crashes. Whilst attempting to save memory to enable E4B to work I attempted: skipping vision tower initialisation (saved 331MB), reduced maxTokens from 2048 to 1024, reduced topK from 5 to 3, increased memory limit (added com.apple.developer.kernel.increased-memory-limit and extended-virtual-addressing entitlement) and attempted KV cache quantisation. Whilst doing I encountered recurrent issues with Swift triggering “EXC_BREAKPOINT” (Swift assertion failure). 

I implemented a debug feature to the app (recording memory at each bootstrap phase, time-to-first-token, token counts, generation duration) – I traced this to an upstream bug in mlx-swift-lm 3.31.3: Gemma 4's attention implementation calls the generic cache.update(keys:values:) method which triggers a fatalError in QuantizedKVCache. This is tracked in ml-explore/mlx-swift-lm PR #237, currently open. I disabled KV cache quantisation as a workaround.  E4B's peak generation memory reached ~6020MB against an under-load ceiling of 5798MB, crashing reproducibly at 5780MB (18MB below the limit). E2B's peak of approximately 4500MB provides 1.3GB of stable headroom. 

Evaluation
A test set of 50 clinical scenarios was written by myself, bespoke to the unit’s clinical guidelines, to evaluate the model’s performance. It contains a range of questions evaluating drug dosing questions, timing/intervals, contact detail retrieval, table queries and questions designed to probe failure modes. 

Both E2B and E4B were tested against the test set, answers were blinded to myself and they were then flagged for clinically dangerous answers (i.e. suggesting an inappropriate dose of medication for a certain gestation), hallucinations and scored 1-3 based on clinical usefulness (3 being useful). An LLM-as-judge solution was tested, using another family of models (qwen3.6:27b) to not introduce self-preference bias – however ultimately, I feel this was not an effective evaluator. One issue, for example, was the LLM judge appeared to give high scores to fluent but clinically dangerous answers. 

The mean quality score of models was comparable across 50 questions, E4B 2.42/3.00, E2B 2.44/3.00. Both models received a score of 3/3 on 31/50 questions, E4B received a score of 1/3 on one additional question (10/50 E4B, 9/50 E2B). Neither model was seen to hallucinate in my test in this evaluation. Both models provided a clinically dangerous answer on one occasion, incorrectly citing an equation for oxygenation index.  

The evaluation tool has been helpful in improving model performance throughout development – the initial model with reduced prompt size, prior to retrieval improvements scored much lower: 2.16/2.32 (E2B/E4B) with more hallucinations (2/1) and more clinically dangerous answers (4/2). The eval framework enabled identification of targets for improvement enabling E2B to overall perform E4B on mean quality, supporting a decision to ship E2B independent of memory constraint.

Limitations
Optimising the app for E2B rather than E4B has lowered the ceiling of the app in the timeframe I had to develop. Moreover, the evaluation set is small at just 50 queries and isn’t robust enough, nor are the results accurate enough, to enable clinical deployment. As a retrieval tool alone, at 97% R@5, there is a genuine clinical use case, however the accuracy of answers generated by the LLM is limited across many use cases at present. 

Code and architecture are open-source (Apache 2.0); the bundled clinical corpus is hospital-specific and not redistributable.

Conclusion
The app was designed to solve real-world friction at low cost, whilst remaining compliant with strict NHS data governance requirements. The ingestion pipeline is designed to be hospital agnostic, allowing replacement of the PDF bundle with any other department/trusts’ guidelines, minor prompt tweaks and then a rebuild of the vector database before shipping to any other site. 

Refusal-by-default is shipped to maximise patient safety, whilst source citation allows for straightforward clinician verification of every answer – and combats LLM mistrust in older generations by providing transparency.

References:
1) Hilton J. Careless costs related to inefficient technology used within NHS England. Clin Med (Lond). 2020 Jan;20(1):115. doi: 10.7861/clinmed.2019-0340. Epub 2019 Nov 8. PMID: 31704730; PMCID: PMC6964190.
2) Zandieh A, Daliri M, Hadian M, Mirrokni V. Turboquant: Online vector quantization with near-optimal distortion rate. arXiv preprint arXiv:2504.19874. 2025 Apr 28.
