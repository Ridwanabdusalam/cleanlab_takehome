# Trustworthiness Detector

A Python library for detecting bad/unreliable outputs from Large Language Models (LLMs) using self-reflection certainty.

Based on the paper: "Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness" (Chen & Mueller, ACL'24).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/your_preference.git
cd your_preference

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your API key and set default model in `.env`:
   ```bash
   # Add your API key (only need one)
   GEMINI_API_KEY=your-actual-key-here
   
   # Set default model
   DEFAULT_MODEL=gemini/gemini-pro
   ```

3. That's it! The library will use your chosen model everywhere.

## Quick Start

```python
from trustworthiness import TrustworthinessDetector

# Uses DEFAULT_MODEL from .env automatically
detector = TrustworthinessDetector()

# Evaluate an answer
score = detector.get_trustworthiness_score(
    question="What is 2 + 2?",
    answer="4"
)
print(f"Trustworthiness: {score}")  # Should be high (~0.9)
```

## Changing Models

To change the model used throughout the library, simply update `DEFAULT_MODEL` in your `.env` file. No code changes needed!

Supported models:
- `gemini/gemini-pro` (recommended - free API)
- `gpt-3.5-turbo`
- `gpt-4`
- `claude-2`
- Any model supported by [litellm](https://github.com/BerriAI/litellm)

## How It Works

The library implements self-reflection certainty:
1. Takes a question-answer pair
2. Asks the LLM to evaluate if its answer is correct
3. Uses multiple follow-up questions for robustness
4. Returns a score between 0 (untrustworthy) and 1 (trustworthy)

## API Reference

### TrustworthinessDetector

```python
detector = TrustworthinessDetector(
    model=None,             # Uses DEFAULT_MODEL if not specified
    temperature=0.0,        # LLM temperature (0 = deterministic)
    cache_responses=True    # Cache to avoid duplicate API calls
)
```

### Main Methods

- `get_trustworthiness_score(question, answer)`: Get score for single Q&A
- `batch_evaluate(qa_pairs)`: Evaluate multiple Q&A pairs efficiently

## Examples

See `examples/usage_example.py` for comprehensive examples.

## Score Interpretation

- **> 0.7**: High confidence (trustworthy)
- **0.3 - 0.7**: Medium confidence (uncertain)
- **< 0.3**: Low confidence (likely incorrect)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/trustworthiness
```

## License

MIT