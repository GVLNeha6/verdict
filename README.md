# Verdict — Evidence-Based AI Claim Verification

Verdict is an AI-powered claim verification platform that combines *evidence retrieval and multi-agent debate* to verify factual claims.

The project is based on *Yilun Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate (ICML 2024)*.

## How It Works

Claim → Evidence Retrieval → 2 AI Agents → 3 Debate Rounds → Judge → Final Verdict

## Experiment Conditions

- *Single LLM* — Direct claim verification
- *Debate Only* — 2 agents with 3 debate rounds
- *Evidence + Debate* — Retrieved evidence with multi-agent debate

## Results

### Verification Results — 50 FEVER Claims

| Method | Accuracy | Precision | Recall | F1-Score |
|:--|--:|--:|--:|--:|
| Single LLM | 60.0% | 42.3% | 58.8% | 48.6% |
| Debate Only | 60.0% | 45.6% | 58.8% | 49.3% |
| **Evidence + Debate** | **74.0%** | **80.6%** | **74.3%** | **73.2%** |

### Evidence Retrieval Results

| Retrieval Method | Hit@1 | Hit@3 | Hit@5 |
|:--|--:|--:|--:|
| TF-IDF | 35.7% | 50.0% | 54.3% |
| BM25 | 35.7% | 51.4% | **57.1%** |
| **Sentence-BERT + FAISS** | **40.0%** | **52.9%** | 55.7% |

## Tech Stack

*Backend:* Python, FastAPI, Sentence Transformers, FAISS

*Frontend:* React, TypeScript, Vite, Tailwind CSS

*Dataset:* FEVER

## Goal

To study whether combining *retrieved evidence with multi-agent debate* can improve the reliability and transparency of AI-based claim verification.
