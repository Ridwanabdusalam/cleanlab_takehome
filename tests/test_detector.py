"""
Test suite for the Trustworthiness Detector.
Tests core functionality and edge cases.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trustworthiness import TrustworthinessDetector, evaluate_trustworthiness, DEFAULT_MODEL
from src.trustworthiness.prompts import REFLECTION_PROMPTS
from src.trustworthiness.config import validate_model_api_key


class TestTrustworthinessDetector:
    """Test cases for TrustworthinessDetector class."""
    
    def test_initialization(self):
        """Test detector initialization with different parameters."""
        # Default initialization
        detector = TrustworthinessDetector()
        # Don't test the specific default model - just that it has one
        assert detector.model is not None
        assert isinstance(detector.model, str)
        assert detector.temperature == 0.0
        assert detector.cache_responses == True
        assert len(detector.reflection_prompts) == 2
        
        # Custom initialization
        detector = TrustworthinessDetector(
            model="gemini/gemini-pro",
            temperature=0.5,
            cache_responses=False
        )
        assert detector.model == "gemini/gemini-pro"
        assert detector.temperature == 0.5
        assert detector.cache_responses == False
        assert detector._cache is None
    
    def test_default_model(self):
        """Test that default model is set correctly."""
        detector = TrustworthinessDetector()
        # Should use the DEFAULT_MODEL from config
        assert detector.model == DEFAULT_MODEL
    
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_model_validation(self):
        """Test model API key validation."""
        # Should work with matching API key
        detector = TrustworthinessDetector(model="gemini/gemini-pro")
        assert detector.model == "gemini/gemini-pro"
        
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with pytest.raises(ValueError, match="API key error"):
            TrustworthinessDetector(model="gemini/gemini-pro")
    
    def test_custom_prompts(self):
        """Test initialization with custom reflection prompts."""
        custom_prompts = [
            "Is {answer} correct for {question}? (A) Yes (B) No (C) Maybe",
            "Double-check: {answer} for {question}? (A) Yes (B) No (C) Maybe"
        ]
        detector = TrustworthinessDetector(reflection_prompts=custom_prompts)
        assert detector.reflection_prompts == custom_prompts
        assert len(detector.reflection_prompts) == 2
    
    @patch('litellm.completion')
    def test_get_trustworthiness_score(self, mock_completion):
        """Test the main scoring function."""
        # Mock LLM responses
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="explanation: The answer is correct. answer: [A]"))]
        mock_completion.return_value = mock_response
        
        detector = TrustworthinessDetector(model="gemini/gemini-pro")
        score = detector.get_trustworthiness_score("What is 2+2?", "4")
        
        # Should be called twice (for 2 reflection prompts)
        assert mock_completion.call_count == 2
        # Score should be 1.0 (both responses are A = correct)
        assert score == 1.0
    
    @patch('litellm.completion')
    def test_mixed_confidence_responses(self, mock_completion):
        """Test with mixed confidence responses."""
        # First call returns "correct", second returns "not sure"
        mock_completion.side_effect = [
            Mock(choices=[Mock(message=Mock(content="answer: [A]"))]),  # Correct
            Mock(choices=[Mock(message=Mock(content="answer: [C]"))]),  # Not sure
        ]
        
        detector = TrustworthinessDetector(model="gemini/gemini-pro")
        score = detector.get_trustworthiness_score("What is 2+2?", "4")
        
        # Score should be (1.0 + 0.5) / 2 = 0.75
        assert score == 0.75
    
    @patch('litellm.completion')
    def test_incorrect_answer_detection(self, mock_completion):
        """Test detection of incorrect answers."""
        # Both responses say incorrect
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="answer: [B]"))]
        mock_completion.return_value = mock_response
        
        detector = TrustworthinessDetector(model="gemini/gemini-pro")
        score = detector.get_trustworthiness_score("What is 2+2?", "5")
        
        # Score should be 0.0 (both responses are B = incorrect)
        assert score == 0.0
    
    def test_parse_reflection_response(self):
        """Test response parsing with various formats."""
        detector = TrustworthinessDetector()
        
        # Test different response formats
        test_cases = [
            ("answer: A", 1.0),
            ("answer: [A]", 1.0),
            ("answer: (A)", 1.0),
            ("explanation: correct. answer: A", 1.0),
            ("Answer: B", 0.0),
            ("answer: [B]", 0.0),
            ("answer: C", 0.5),
            ("answer: [C]", 0.5),
            ("The answer is A", 0.5),  # Can't parse, default to uncertain
            ("", 0.5),  # Empty response
            ("random text", 0.5),  # Unparseable
        ]
        
        for response, expected_score in test_cases:
            score = detector._parse_reflection_response(response)
            assert score == expected_score, f"Failed for response: {response}"
    
    @patch('litellm.completion')
    def test_caching_functionality(self, mock_completion):
        """Test that caching works correctly."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="answer: [A]"))]
        mock_completion.return_value = mock_response
        
        detector = TrustworthinessDetector(
            model="gemini/gemini-pro",
            cache_responses=True
        )
        
        # First call
        score1 = detector.get_trustworthiness_score("What is 2+2?", "4")
        initial_call_count = mock_completion.call_count
        
        # Second call with same Q&A
        score2 = detector.get_trustworthiness_score("What is 2+2?", "4")
        
        # Should use cache, not make new API calls
        assert mock_completion.call_count == initial_call_count
        assert score1 == score2
        assert len(detector._cache) > 0
    
    @patch('litellm.completion')
    def test_batch_evaluate(self, mock_completion):
        """Test batch evaluation functionality."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="answer: [A]"))]
        mock_completion.return_value = mock_response
        
        detector = TrustworthinessDetector(model="gemini/gemini-pro")
        
        qa_pairs = [
            ("What is 2+2?", "4"),
            ("What is 3+3?", "6"),
            ("What is 4+4?", "8"),
        ]
        
        scores = detector.batch_evaluate(qa_pairs, show_progress=False)
        
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(0 <= s <= 1 for s in scores)
    
    @patch('litellm.completion')
    def test_error_handling(self, mock_completion):
        """Test error handling when LLM fails."""
        # Simulate API error
        mock_completion.side_effect = Exception("API Error")
        
        detector = TrustworthinessDetector(model="gemini/gemini-pro")
        score = detector.get_trustworthiness_score("What is 2+2?", "4")
        
        # Should return 0.5 (uncertain) when API fails
        assert score == 0.5
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        detector = TrustworthinessDetector(cache_responses=True)
        
        # Add some dummy cache entries
        detector._cache["test1"] = 0.8
        detector._cache["test2"] = 0.9
        
        assert len(detector._cache) == 2
        
        detector.clear_cache()
        
        assert len(detector._cache) == 0
    
    def test_no_cache_mode(self):
        """Test that caching can be disabled."""
        detector = TrustworthinessDetector(cache_responses=False)
        
        assert detector._cache is None
        
        # Should not crash when trying to use cache operations
        detector.clear_cache()  # Should do nothing


class TestConvenienceFunction:
    """Test the convenience function."""
    
    @patch('litellm.completion')
    def test_evaluate_trustworthiness(self, mock_completion):
        """Test the standalone evaluation function."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="answer: [A]"))]
        mock_completion.return_value = mock_response
        
        score = evaluate_trustworthiness("What is 2+2?", "4", model="gemini/gemini-pro")
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert mock_completion.call_count == 2  # Two reflection prompts


class TestPrompts:
    """Test prompt templates."""
    
    def test_prompt_templates(self):
        """Test that prompt templates are properly formatted."""
        assert len(REFLECTION_PROMPTS) == 2
        
        # Test that prompts contain required placeholders
        for prompt in REFLECTION_PROMPTS:
            assert "{question}" in prompt
            assert "{answer}" in prompt
            assert "(A)" in prompt
            assert "(B)" in prompt
            assert "(C)" in prompt
    
    def test_prompt_formatting(self):
        """Test that prompts can be formatted correctly."""
        question = "What is 2+2?"
        answer = "4"
        
        for prompt in REFLECTION_PROMPTS:
            formatted = prompt.format(question=question, answer=answer)
            assert question in formatted
            assert answer in formatted
            assert "choose one letter from among choices A through C" in formatted


@pytest.mark.integration
class TestIntegration:
    """Integration tests (requires API key)."""
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
    def test_real_api_call(self):
        """Test with actual API call to Gemini."""
        detector = TrustworthinessDetector(model="gemini/gemini-pro")
        
        # Test with obviously correct answer
        score = detector.get_trustworthiness_score("What is 1+1?", "2")
        assert score > 0.7  # Should have high confidence
        
        # Test with obviously wrong answer
        score = detector.get_trustworthiness_score("What is 1+1?", "3")
        assert score < 0.3  # Should have low confidence


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])