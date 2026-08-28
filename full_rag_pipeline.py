# Week 20 Day 3-4: Full RAG Pipeline (fixed)
import re, os, time
import numpy as np

DIRS = ["/mnt/workspace/.cache/modelscope","/mnt/data/.cache/modelscope",
        os.path.expanduser("~/.cache/modelscope")]

def find_model(name):
    s = [name, name.replace(".","___"), name.lower()]
    for r in DIRS:
        if not os.path.isdir(r): continue
        for rd, dd, ff in os.walk(r):
            for d in dd:
                fp = os.path.join(rd,d)
                for n in s:
                    if n in fp and os.path.exists(os.path.join(fp,"config.json")):
                        return fp
    from modelscope import snapshot_download
    return snapshot_download(name)


def load_embedder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    mp = find_model("BAAI/bge-small-zh-v1.5")
    print("   Embedder:", mp)
    tok = AutoTokenizer.from_pretrained(mp)
    mdl = AutoModel.from_pretrained(mp)
    gpu = torch.cuda.is_available()
    if gpu: mdl = mdl.to("cuda"); print("   GPU")
    else: print("   CPU")
    mdl.eval()

    def enc(txts, bs=16):
        res = []; n = len(txts)
        for i in range(0, n, bs):
            b = txts[i:i+bs]
            inp = tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt")
            if gpu: inp = {k:v.to("cuda") for k,v in inp.items()}
            with torch.no_grad():
                out = mdl(**inp)
                h = out.last_hidden_state
                m = inp["attention_mask"].unsqueeze(-1).float()
                p = (h*m).sum(dim=1)/m.sum(dim=1)
                res.append(p.cpu().numpy())
            d = min(i+bs, n); pct = 100*d//n
            bar = "#"*(pct//4)+"-"*(25-pct//4)
            print("\r   emb [%s] %d/%d"%(bar,d,n), end="", flush=True)
        print()
        e = np.concatenate(res, axis=0)
        return e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-8)
    return enc


def load_reranker():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    mp = find_model("BAAI/bge-reranker-base")
    print("   Reranker:", mp)
    tok = AutoTokenizer.from_pretrained(mp)
    mdl = AutoModelForSequenceClassification.from_pretrained(mp)
    gpu = torch.cuda.is_available()
    if gpu: mdl = mdl.to("cuda"); print("   GPU")
    else: print("   CPU")
    mdl.eval()

    def rank(query, docs):
        pairs = [[query, d] for d in docs]
        scores = []
        bs = 16
        for i in range(0, len(pairs), bs):
            batch = pairs[i:i+bs]
            inp = tok(batch, padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            if gpu: inp = {k:v.to("cuda") for k,v in inp.items()}
            with torch.no_grad():
                out = mdl(**inp, return_dict=True)
                s = out.logits.view(-1).cpu().numpy()
                scores.extend(s.tolist())
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return ranked
    return rank


def load_generator():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mp = find_model("Qwen2.5-0.5B-Instruct")
    print("   Generator:", mp)
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(mp, trust_remote_code=True,
                                                torch_dtype=torch.float16)
    gpu = torch.cuda.is_available()
    if gpu: mdl = mdl.to("cuda"); print("   GPU")
    else: print("   CPU")
    mdl.eval()

    def _generate(prompt, max_tokens=300):
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if gpu: inp = {k:v.to("cuda") for k,v in inp.items()}
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=max_tokens,
                               temperature=0.3, do_sample=True,
                               pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True)
        return text.strip()

    def gen_with_rag(query, contexts, max_tokens=300):
        ctx_text = "\n\n---\n\n".join(contexts[:3])
        prompt = (
            "Based on the following reference documents, answer the question.\n"
            "If the documents do not contain enough information, say so.\n\n"
            "Reference documents:\n%s\n\n"
            "Question: %s\n\n"
            "Answer:" % (ctx_text, query)
        )
        return _generate(prompt, max_tokens)

    def gen_without_rag(query, max_tokens=300):
        prompt = "Question: %s\n\nAnswer:" % query
        return _generate(prompt, max_tokens)

    class GenDispatch:
        def __call__(self, query, contexts=None, max_tokens=300):
            if contexts is None:
                return gen_without_rag(query, max_tokens)
            return gen_with_rag(query, contexts, max_tokens)

    return GenDispatch()


class RAGPipeline:
    def __init__(self, doc_text):
        self.doc = doc_text
        self.chunks = self._chunk(doc_text)

    def _chunk(self, txt, sz=200, ol=40):
        w = txt.split(); step = max(1,sz-ol)
        return [" ".join(w[i:i+sz]) for i in range(0,len(w),step)
                if " ".join(w[i:i+sz]).strip() and i+sz<len(w)+step]

    def build_index(self, encode_fn):
        print("   Building index: %d chunks..." % len(self.chunks))
        self.embs = encode_fn(self.chunks)
        import faiss
        self.index = faiss.IndexFlatIP(self.embs.shape[1])
        self.index.add(self.embs.astype(np.float32))
        print("   Index ready: %d vectors" % self.index.ntotal)

    def search(self, query_emb, top_k=20):
        scores, ids = self.index.search(query_emb.astype(np.float32).reshape(1,-1), top_k)
        return [(ids[0][i], scores[0][i], self.chunks[ids[0][i]])
                for i in range(len(ids[0])) if ids[0][i]>=0]

    def answer(self, query, encode_fn, rerank_fn, gen_fn, use_retrieval=True, use_rerank=True):
        print("\n   Q: %s" % query)
        t0 = time.time()

        if not use_retrieval:
            ans = gen_fn(query)
            t = time.time()-t0
            print("   No retrieval -> generated (%.1fs)" % t)
            return {"query": query, "answer": ans, "method": "LLM-only",
                    "time": t, "contexts": []}

        qe = encode_fn([query])[0]
        hits = self.search(qe, top_k=20)
        t1 = time.time()-t0

        if use_rerank and rerank_fn:
            docs = [h[2] for h in hits]
            ranked = rerank_fn(query, docs)
            top_docs = [d for _, d in ranked[:3]]
            t2 = time.time()-t0
        else:
            top_docs = [h[2] for h in hits[:3]]
            t2 = t1

        ans = gen_fn(query, top_docs)
        t3 = time.time()-t0

        print("   Search: %.1fs | Rerank: %.1fs | Generate: %.1fs | Total: %.1fs" % (
            t1, t2-t1, t3-t2, t3))

        return {"query": query, "answer": ans,
                "method": "RAG%s" % ("+rerank" if use_rerank else ""),
                "time": t3, "search_time": t1, "rerank_time": t2-t1,
                "gen_time": t3-t2, "contexts": top_docs}


def demo():
    doc = """
    Transfer learning is a machine learning technique where a model developed for one task
    is reused as the starting point for a model on a second task. In NLP, pre-trained models
    like BERT and GPT have revolutionized the field by reducing the need for large labeled datasets.
    Fine-tuning involves taking a pre-trained model and training it further on a specific downstream
    task with a smaller learning rate.

    The Transformer architecture, introduced in the paper "Attention Is All You Need" in 2017,
    uses self-attention mechanisms to process all tokens in parallel. Unlike RNNs which process
    sequences step by step, transformers can capture long-range dependencies more effectively.
    The key components include multi-head attention, positional encoding, and feed-forward networks.

    Attention mechanisms compute weighted sums of values based on similarity between queries and keys.
    Self-attention is a special case where Q, K, V all come from the same sequence. The formula
    is: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. This allows each token to attend to all
    other tokens in the sequence simultaneously.

    RAG (Retrieval-Augmented Generation) combines retrieval systems with language models.
    Instead of relying only on model parameters, RAG retrieves relevant documents from an external
    knowledge base and conditions generation on both the query and retrieved documents.
    This reduces hallucination and enables using up-to-date information.

    FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and
    clustering of dense vectors. It supports various index types: Flat (exact search),
    IVF (inverted file with clustering), HNSW (hierarchical navigable small world graphs),
    and PQ (product quantization for compression). The choice depends on the trade-off
    between speed, accuracy, and memory usage.

    RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a
    hierarchical tree structure over documents. It recursively clusters document chunks
    and generates summaries at each level, enabling retrieval at multiple granularities.
    However, it works best for multi-topic documents and is less beneficial for single
    linear papers.
    """

    print("=" * 60)
    print("Full RAG Pipeline Demo")
    print("=" * 60)

    print("\n[1/3] Load Embedder (BGE-small-zh-v1.5)")
    t0 = time.time(); enc = load_embedder()
    print("   %.1fs" % (time.time()-t0))

    print("\n[2/3] Load Reranker (bge-reranker-base)")
    t0 = time.time()
    try:
        rrk = load_reranker(); print("   %.1fs" % (time.time()-t0))
    except Exception as e:
        print("   SKIP: %s" % e); rrk = None

    print("\n[3/3] Load Generator (Qwen2.5-0.5B-Instruct)")
    t0 = time.time(); gen = load_generator()
    print("   %.1fs" % (time.time()-t0))

    print("\n--- Build Index ---")
    rag = RAGPipeline(doc)
    rag.build_index(enc)

    queries = [
        "What is transfer learning and how is it used in NLP?",
        "How does the attention mechanism work in transformers?",
        "What is RAG and why is it useful?",
    ]

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: LLM-only (no retrieval)")
    print("=" * 60)
    for q in queries:
        r = rag.answer(q, enc, rrk, gen, use_retrieval=False)
        print("   >>>> %s" % r["answer"][:200])

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: RAG (retrieval only, no rerank)")
    print("=" * 60)
    for q in queries:
        r = rag.answer(q, enc, rrk, gen, use_retrieval=True, use_rerank=False)
        print("   >>>> %s" % r["answer"][:200])

    if rrk:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: RAG + Rerank (full pipeline)")
        print("=" * 60)
        for q in queries:
            r = rag.answer(q, enc, rrk, gen, use_retrieval=True, use_rerank=True)
            print("   >>>> %s" % r["answer"][:200])

    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN (RAG + Rerank per query)")
    print("=" * 60)
    use_rrk = rrk is not None
    total = 0
    for q in queries:
        r = rag.answer(q, enc, rrk, gen, use_retrieval=True, use_rerank=use_rrk)
        total += r["time"]
    print("\n   Avg per query: %.1fs" % (total/len(queries)))

if __name__ == "__main__":
    demo()
