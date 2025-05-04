# Environment

Python: 3.10

CUDA: 12.2

GPU: 2 x A800 (80GB)

This will create a conda environment called `infl`:
```bash
conda env create -f environment
```


# Experiments

## Exp 1: Scalability validation of Inf-EKFAC

Code, intermediate results, trained models and final are all in `1_scalability_validation/`.

1. Train a toy model.
    
    Prepare `ptb_text_only` dataset. We use the version from Huggingface.

    Run `train.py`. This results in a model checkpoint `model.ckpt` and model architecture specification `config.json`.
2. Sample query/candidate datapoints.

    Run "Sample influence estimation targets and candidates" section of `test.ipynb` to obtain `choices_queries.pkl` and `choices_candidates.pkl`.
3. Run influence estimation baselines. To do this we use a manually modified (with forward/backward hooks) version of GPT-NeoX architecture in `modeling_gpt_neox.py`.

    Figures showing the convergence trend are kept in `figures/` and influence estimates are kept in `infls/`. Results are **manually** entered into `results.json`.

    - Conjugate Gradient: `cg.py`
    - Inf-EKFAC: `ekfac.py`
    - LiSSA: `lissa.py`
    - GDP: `gradient_dot_product.py`
    - CKA: `cka.py`
    
    TRAK: Things are a little more complicated.
    
    First train models on random subsets using `trak-train.py`. The models are kept in `trak_models/`.
    
    Then run `trak-infl.py` to obtain influence estimates.
4. Plot results.
    Use code in "Plotting" section of `test.ipynb` to draw and save plots using results saved in `results.json`.

## Exp 2: Multi-stage influence benchmark

Code is in `3_multi_stage_benchmark`.

1. Prepare data.

    Download full JSON format of TREx from https://hadyelsahar.github.io/t-rex/downloads/, and extract the ZIP file to `data/TREX/`.

    Download LAMA from https://dl.fbaipublicfiles.com/LAMA/data.zip, and extract the ZIP to `data/LAMA/`.

    Run `data/prepare_trex.py` to obtain a sentence-level attribution set in `data/HF_TREX_sentences/`.

    Copy `data/bloom_questions.jsonl` into `data/LAMA/bloom_questions.jsonl`.

    Run `data/facts_to_ids.py` to obtain `data/fact_to_ids.json`.

    Run "Test data" section in `trace/test.ipynb` to obtain `trace/test_data_selections.json`.

    Download `roots_en_wikipedia` and `xP3` dataset from Huggingface.
2. Prepare model.

    bloom-560m and bloomz-560m from Huggingface.
3. Run BM25 baseline.

    `trace/bm25.py`.
4. Get reranker candidates.

    After running BM25 baseline, run "Reranker candidates" section in `trace/test.ipynb` to obtain `trace/target_sharing_ids.json`.

    Run `trace/reranker_prepare.py` to obtain `reranker_candidates.json`.
5. Multi-stage influence.

    Run `ekfac/fit-pretrain.py` and `ekfac/fit-finetune.py`.

    Run `influence_two_stages.py`. Results in `trace/bloom_two_stages_infl_candidates/`.
6. Embedding-similarity baseline.

    Run `ekfac/influence_embed.py`. Results in `trace/bloom_embed_sim_candidates/`.
7. Metrics.

    MRR metric in `trace/mrr.py`. Recall metric in `trace/recall.py`.
8. LAMA wikidata entity ids overlap ratio with `roots_en_wikipedia`.

    `roots_en_wikipedia.py`.
9. Top-k accuracy on LAMA.

    `evidences.py` for "evidence" field of LAMA.

    `question.py` for using `data/bloom_questions.jsonl` template.

    `topk-accuracy.py` for using original LAMA template.


## Exp 3: Case study

Code in `4_case_study`.

1. Prepare data.

    Download `monology/pile-uncopyrighted` and `databricks/databricks-dolly-15k` dataset from Huggingface.

    Use first 10% of the first file of `pile-uncopyrighted` and save in `small-trainset`, and tokenize it using `tokenize_dataset.py` to `tokenized_ds.pt`.

    Tokenize `databricks/databricks-dolly-15k` using `dollybricks-15k` section of `test.ipynb`.

    Download `20200301-en` version of Wikipedia dataset from Tensorflow Datasets.
2. Prepare model.

    Download `EleutherAI/pythia-2.3b` and `databricks/dolly-v2-3b` from Huggingface.

    Download `all-MiniLM-L6-v2` from Huggingface, prepared for building KNN.
3. Build KNN.

    Run `knn.py` for bias attribution and `knn-wikipedia.py` for fact tracing. Results in `knn.joblib` and `knn-wikipedia.joblib`.
4. Fit EK-FAC factors.

    Run `ekfac_fit.py`. Intermediate results in `factors/`.
5. Compute multi-stage influence of candidates.

    Run `influence.py`. Intermediate results in `infls/`.
6. Retrieve top influential training candidates.

    Run `top-infl-samples.py`. Results in `results/` in json format.
