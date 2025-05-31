from transformers import pipeline
from cnn_dailymail_subset import load_small_subset
from tqdm import tqdm

def run_summarization(model_name="facebook/bart-base", max_samples=50, num_beams=4):
  """
  Generates abstractive summaries using a pretrained model with optional beam search.

  Args:
    model_name (str): HuggingFace model name (e.g., 'facebook/bart-base', 't5-small').
    max_samples (int): Number of examples to summarize.
    num_beams (int): Beam width for beam search decoding.

  Returns:
    List of tuples: (article, generated_summary, reference_summary)
  """
  print(f"Loading dataset subset with {max_samples} samples...")
  data = load_small_subset(num_samples=max_samples)

  print(f"Loading summarization model: {model_name}")
  summarizer = pipeline("summarization", model=model_name)

  print("Generating summaries...")
  summaries = []
  for item in tqdm(data, desc="Summarizing"):
    article = item["article"]
    reference = item["highlights"]

    # Truncate long articles to avoid token overflow
    article = article[:3500]

    # For T5, prefix input with "summarize: "
    if "t5" in model_name:
      article = "summarize: " + article

    summary = summarizer(
      article,
      max_new_tokens=130,
      min_length=30,
      do_sample=False,
      num_beams=num_beams
    )[0]["summary_text"]

    summaries.append((article, summary, reference))

  return summaries
