# Abstractive Summarization of News Articles

This project implements an **abstractive text summarization** system using the `facebook/bart-base` transformer model from Hugging Face. It generates human-like summaries of news articles and evaluates them using standard ROUGE metrics.

## Features

- Uses pretrained BART model for abstractive summarization
- Works with a subset of the CNN/DailyMail dataset
- Includes ROUGE-1, ROUGE-2, and ROUGE-L evaluation
- Implements beam search decoding for improved fluency
- Clean modular structure for easy experimentation

## Project Structure

```
abstractive-summarization/
├── cnn_dailymail_subset.py    # Loads a small test subset
├── summarizer.py              # Runs the summarizer (with beam search)
├── evaluate.py                # Computes ROUGE scores and shows outputs
├── main.py                    # Entry point for running everything
├── results/
│   └── sample_outputs.txt     # Generated summaries and references
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/AndrewChan8/abstractive-summarization.git
cd abstractive-summarization
pip install -r requirements.txt
```

### 2. Run the summarization pipeline

```bash
python3 main.py
```

This will:
- Load 10 samples from CNN/DailyMail
- Generate summaries with `facebook/bart-base` using beam search
- Print ROUGE scores and show 3 sample outputs
- Save results to `results/sample_outputs.txt`

## Sample ROUGE Scores (10 Samples)

| Metric   | Score   |
|----------|---------|
| ROUGE-1  | 0.3505  |
| ROUGE-2  | 0.1389  |
| ROUGE-L  | 0.2254  |

## Sample Output

See `results/sample_outputs.txt` for a few full-length examples of:
- Original article
- Generated summary
- Reference (human-written) summary

## Future Improvements

- Fine-tune BART or use domain-adapted models like PEGASUS
- Experiment with T5, top-k sampling, or nucleus sampling
- Use a larger subset for better metric stability

