"""
Trustworthiness Detector for LLM outputs
Implements self-reflection certainty from BSDetector paper
"""
import re
from typing import List, Tuple, Optional
import litellm
from .config import DEFAULT_MODEL, validate_model_api_key


class TrustworthinessDetector:
    """
    Detects trustworthiness of LLM answers using self-reflection certainty.
    
    Based on "Quantifying Uncertainty in Answers from any Language Model 
    and Enhancing their Trustworthiness" (Chen & Mueller, ACL'24)
    """
    
    # Default reflection prompts from the paper (Figure 6b)
    DEFAULT_REFLECTION_PROMPTS = [
        """Question: {question}
Proposed Answer: {answer}
Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template: 
explanation: [insert analysis], answer: [choose one letter from among choices A through C]""",
        
        """Question: {question}
Proposed Answer: {answer}
Are you really sure the proposed answer is correct? 
Choose again: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template: 
explanation: [insert analysis], answer: [choose one letter from among choices A through C]"""
    ]
    
    def __init__(
        self, 
        model: str = None,
        reflection_prompts: Optional[List[str]] = None,
        temperature: float = 0.0,
        cache_responses: bool = True
    ):
        """
        Initialize the trustworthiness detector.
        
        Args:
            model: LLM model to use (via litellm). If None, uses DEFAULT_MODEL from config
            reflection_prompts: Custom reflection prompts (uses defaults if None)
            temperature: Temperature for LLM responses (0 for deterministic)
            cache_responses: Whether to cache reflection responses
        """
        # Use configured default if no model specified
        self.model = model or DEFAULT_MODEL
        
        # Validate API key for chosen model
        is_valid, message = validate_model_api_key(self.model)
        if not is_valid:
            raise ValueError(f"API key error: {message}")
        
        self.reflection_prompts = reflection_prompts or self.DEFAULT_REFLECTION_PROMPTS
        self.temperature = temperature
        self.cache_responses = cache_responses
        self._cache = {} if cache_responses else None
        
    def get_trustworthiness_score(self, question: str, answer: str) -> float:
        """
        Calculate trustworthiness score for a question-answer pair.
        
        Args:
            question: The original question
            answer: The answer to evaluate
            
        Returns:
            Trustworthiness score between 0 and 1
        """
        reflection_scores = self._get_self_reflection_scores(question, answer)
        return sum(reflection_scores) / len(reflection_scores)
    
    def _get_self_reflection_scores(self, question: str, answer: str) -> List[float]:
        """Get scores from multiple self-reflection prompts."""
        scores = []
        
        for i, prompt_template in enumerate(self.reflection_prompts):
            # Format the prompt
            prompt = prompt_template.format(question=question, answer=answer)
            
            # Check cache if enabled
            cache_key = f"{question}|{answer}|{i}"
            if self.cache_responses and cache_key in self._cache:
                score = self._cache[cache_key]
            else:
                # Query LLM
                response = self._query_llm(prompt)
                
                # Parse response to get score
                score = self._parse_reflection_response(response)
                
                # Cache if enabled
                if self.cache_responses:
                    self._cache[cache_key] = score
            
            scores.append(score)
            
        return scores
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with error handling."""
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Warning: LLM query failed: {e}")
            # Return a response that will be parsed as uncertain
            return "answer: [C]"
    
    def _parse_reflection_response(self, response: str) -> float:
        """
        Parse LLM response to extract choice and convert to score.
        
        Returns:
            1.0 for (A) Correct
            0.0 for (B) Incorrect  
            0.5 for (C) I am not sure or parsing failure
        """
        # Look for the answer pattern
        # Handles formats like "answer: A", "answer: [A]", "answer: (A)"
        pattern = r'answer:\s*[\[\(]?([ABC])[\]\)]?'
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            choice = match.group(1).upper()
            score_map = {'A': 1.0, 'B': 0.0, 'C': 0.5}
            return score_map.get(choice, 0.5)
        else:
            # If we can't parse, default to uncertain
            print(f"Warning: Could not parse response: {response[:100]}...")
            return 0.5
    
    def batch_evaluate(
        self, 
        qa_pairs: List[Tuple[str, str]], 
        show_progress: bool = True
    ) -> List[float]:
        """
        Evaluate multiple question-answer pairs.
        
        Args:
            qa_pairs: List of (question, answer) tuples
            show_progress: Whether to show progress
            
        Returns:
            List of trustworthiness scores
        """
        scores = []
        
        for i, (question, answer) in enumerate(qa_pairs):
            if show_progress:
                print(f"Evaluating {i+1}/{len(qa_pairs)}...", end='\r')
            
            score = self.get_trustworthiness_score(question, answer)
            scores.append(score)
        
        if show_progress:
            print(f"Evaluated {len(qa_pairs)} Q&A pairs.    ")
            
        return scores
    
    def clear_cache(self):
        """Clear the response cache."""
        if self._cache is not None:
            self._cache.clear()


# Convenience function for quick evaluation
def evaluate_trustworthiness(
    question: str, 
    answer: str, 
    model: str = None
) -> float:
    """
    Quick function to evaluate a single Q&A pair.
    
    Args:
        question: The question
        answer: The answer to evaluate
        model: LLM model to use (if None, uses DEFAULT_MODEL from config)
        
    Returns:
        Trustworthiness score between 0 and 1
    """
    detector = TrustworthinessDetector(model=model)
    return detector.get_trustworthiness_score(question, answer)
