from summarizer import run_summarization
from evaluate import evaluate_rouge, print_sample_outputs

if __name__ == "__main__":
  model = "facebook/bart-base"
  max_samples = 10

  print("Starting summarization...")
  summaries = run_summarization(model_name="facebook/bart-base", max_samples=10, num_beams=4)

  print("\nEvaluating summaries...")
  evaluate_rouge(summaries)

  print_sample_outputs(summaries, num_samples=3)

  # Save all summaries to a text file
  with open("results/sample_outputs.txt", "w") as f:
    for i, (article, generated, reference) in enumerate(summaries):
      f.write(f"Sample {i+1}\n")
      f.write(f"{'-'*40}\n")
      f.write(f"ARTICLE:\n{article}\n\n")
      f.write(f"GENERATED SUMMARY:\n{generated}\n\n")
      f.write(f"REFERENCE SUMMARY:\n{reference}\n")
      f.write(f"{'='*60}\n\n")
