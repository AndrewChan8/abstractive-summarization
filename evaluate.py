from rouge_score import rouge_scorer
import numpy as np

def evaluate_rouge(summaries):
  """
  Computes average ROUGE-1, ROUGE-2, and ROUGE-L scores.

  Args:
    summaries (list): List of (article, generated_summary, reference_summary) tuples
  """
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

  r1_scores, r2_scores, rl_scores = [], [], []

  for _, generated, reference in summaries:
    scores = scorer.score(reference, generated)
    r1_scores.append(scores['rouge1'].fmeasure)
    r2_scores.append(scores['rouge2'].fmeasure)
    rl_scores.append(scores['rougeL'].fmeasure)

  print("\nAverage ROUGE Scores:")
  print(f"ROUGE-1: {np.mean(r1_scores):.4f}")
  print(f"ROUGE-2: {np.mean(r2_scores):.4f}")
  print(f"ROUGE-L: {np.mean(rl_scores):.4f}")

def print_sample_outputs(summaries, num_samples=5):
  """
  Prints a few samples of article, generated summary, and reference.

  Args:
    summaries (list): List of (article, generated_summary, reference_summary) tuples
    num_samples (int): Number of samples to print
  """
  print("\n--- Sample Summaries ---")
  for i, (article, generated, reference) in enumerate(summaries[:num_samples]):
    print(f"\nSample {i + 1}")
    print(f"{'-'*40}\nARTICLE:\n{article[:500]}...")
    print(f"\nGENERATED SUMMARY:\n{generated}")
    print(f"\nREFERENCE SUMMARY:\n{reference}")
    print(f"{'='*60}")
