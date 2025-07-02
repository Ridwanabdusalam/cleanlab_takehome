"""
Example usage of the Trustworthiness Detector library.
Shows how to evaluate LLM answers for trustworthiness.
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trustworthiness import TrustworthinessDetector, DEFAULT_MODEL, validate_model_api_key
import litellm

# Load environment variables
load_dotenv()


def main():
    """Demonstrate the trustworthiness detector with examples from the assignment."""
    
    # Validate API key for default model
    is_valid, message = validate_model_api_key()
    if not is_valid:
        print(f"ERROR: {message}")
        print("\nTo fix this, either:")
        print("1. Set the appropriate API key in your .env file")
        print("2. Change DEFAULT_MODEL in .env to match your available API key")
        return
    
    print(f"Using model: {DEFAULT_MODEL}")
    print("=" * 50)
    
    # Initialize detector - will use DEFAULT_MODEL automatically
    detector = TrustworthinessDetector(
        temperature=0.0,
        cache_responses=True
    )
    
    print("\n=== Trustworthiness Detector Demo ===")
    print("Testing self-reflection certainty implementation\n")
    
    # Test 1: Examples from the assignment brief
    print("TEST 1: Assignment Examples")
    print("-" * 40)
    
    assignment_tests = [
        # These are the exact examples from the Cleanlab assignment
        {
            "question": "What is 1 + 1?",
            "answer": "2",
            "expected": "high",
            "reason": "Simple correct answer"
        },
        {
            "question": "what is the third month in alphabetical order",
            "answer": "April",  # April, August, December...
            "expected": "high",
            "reason": "Correct factual answer"
        },
        {
            "question": 'How many syllables are in the following phrase: "How much wood could a woodchuck chuck if a woodchuck could chuck wood"? Answer with a single number only.',
            "answer": "14",
            "expected": "high",
            "reason": "Correct count"
        },
    ]
    
    for test in assignment_tests:
        score = detector.get_trustworthiness_score(test["question"], test["answer"])
        status = get_status_symbol(score)
        print(f"{status} Q: {test['question'][:60]}...")
        print(f"  A: {test['answer']}")
        print(f"  Score: {score:.3f} - {test['reason']}")
        print()
    
    # Test 2: Wrong answers (should get low scores)
    print("\nTEST 2: Wrong Answers")
    print("-" * 40)
    
    wrong_answer_tests = [
        {
            "question": "What is 1 + 1?",
            "answer": "3",
            "correct_answer": "2"
        },
        {
            "question": "what is the third month in alphabetical order",
            "answer": "March",
            "correct_answer": "April"
        },
        {
            "question": "What is the capital of France?",
            "answer": "London",
            "correct_answer": "Paris"
        },
    ]
    
    for test in wrong_answer_tests:
        score = detector.get_trustworthiness_score(test["question"], test["answer"])
        status = get_status_symbol(score)
        print(f"{status} Q: {test['question']}")
        print(f"  Wrong A: {test['answer']} (correct: {test['correct_answer']})")
        print(f"  Score: {score:.3f}")
        print()
    
    # Test 3: Uncertain/ambiguous cases
    print("\nTEST 3: Uncertain Cases")
    print("-" * 40)
    
    uncertain_tests = [
        "What will be the most popular programming language in 2030?",
        "What will the weather be like next month?",
        "Who will win the next election?",
    ]
    
    for question in uncertain_tests:
        # Generate an answer for uncertain question
        response = litellm.completion(
            model="gemini/gemini-pro",
            messages=[{"role": "user", "content": question}],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        
        score = detector.get_trustworthiness_score(question, answer)
        status = get_status_symbol(score)
        print(f"{status} Q: {question}")
        print(f"  A: {answer[:100]}...")
        print(f"  Score: {score:.3f} (expected: medium confidence)")
        print()
    
    # Test 4: Batch evaluation
    print("\nTEST 4: Batch Evaluation")
    print("-" * 40)
    
    qa_pairs = [
        # Mix of correct and incorrect answers
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("What is the largest planet in our solar system?", "Earth"),
        ("How many legs does a spider have?", "8"),
        ("How many legs does a spider have?", "6"),
        ("What year did World War II end?", "1945"),
        ("What year did World War II end?", "1942"),
        ("What is the chemical formula for water?", "H2O"),
        ("What is the chemical formula for water?", "CO2"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("Who painted the Mona Lisa?", "Pablo Picasso"),
    ]
    
    print("Evaluating mix of correct and incorrect answers...")
    scores = detector.batch_evaluate(qa_pairs, show_progress=True)
    
    print("\nResults:")
    correct_count = 0
    for (q, a), score in zip(qa_pairs, scores):
        status = get_status_symbol(score)
        confidence = get_confidence_level(score)
        if score > 0.7:
            correct_count += 1
        print(f"{status} {q[:40]}... → {a[:20]:<20} Score: {score:.3f} ({confidence})")
    
    print(f"\nSummary: {correct_count}/{len(qa_pairs)} identified as trustworthy")
    
    # Show performance metrics
    print("\n=== Performance Summary ===")
    print(f"Cache hits: {len(detector._cache) if detector.cache_responses else 0}")
    print(f"Score distribution:")
    print(f"  High confidence (>0.7): {sum(1 for s in scores if s > 0.7)}")
    print(f"  Low confidence (<0.3): {sum(1 for s in scores if s < 0.3)}")
    print(f"  Uncertain (0.3-0.7): {sum(1 for s in scores if 0.3 <= s <= 0.7)}")


def real_world_example():
    """Show a real-world integration example."""
    print("\n=== Real-World Integration Example ===")
    print("Generating and evaluating answers in a typical use case\n")
    
    detector = TrustworthinessDetector(model="gemini/gemini-pro")
    
    # Simulate a Q&A system
    questions = [
        "What is the speed of light in vacuum?",
        "Explain quantum entanglement in simple terms.",
        "What are the main causes of climate change?",
    ]
    
    for question in questions:
        print(f"Question: {question}")
        
        # Generate answer
        response = litellm.completion(
            model="gemini/gemini-pro",
            messages=[{"role": "user", "content": question}],
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
        
        # Evaluate trustworthiness
        score = detector.get_trustworthiness_score(question, answer)
        
        print(f"Answer: {answer[:150]}...")
        print(f"Trustworthiness: {score:.3f}")
        
        # Decision logic
        if score > 0.7:
            print("✓ Answer approved - high confidence\n")
        elif score < 0.3:
            print("✗ Answer rejected - low confidence, regenerating...\n")
            # In real app, you might regenerate or ask human
        else:
            print("? Answer flagged for review - medium confidence\n")


def get_status_symbol(score):
    """Return appropriate symbol based on score."""
    if score > 0.7:
        return "✓"
    elif score < 0.3:
        return "✗"
    else:
        return "?"


def get_confidence_level(score):
    """Return confidence level description."""
    if score > 0.7:
        return "high confidence"
    elif score < 0.3:
        return "low confidence"
    else:
        return "medium confidence"


if __name__ == "__main__":
    main()
    
    # see real-world integration example
    real_world_example()
