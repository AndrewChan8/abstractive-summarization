from datasets import load_dataset
import random

def load_small_subset(num_samples=100, split="test", seed=42):
  """
  Loads a small subset of the CNN/DailyMail dataset for summarization.

  Args:
    num_samples (int): Number of samples to load.
    split (str): Dataset split to use ('train', 'validation', 'test').
    seed (int): Random seed for reproducibility.

  Returns:
    A HuggingFace Dataset object with num_samples examples.
  """
  dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
  return dataset.shuffle(seed=seed).select(range(num_samples))
