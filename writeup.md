# Development Process Write-up

## Overview

I implemented a Python library that detects unreliable LLM outputs using the self-reflection certainty algorithm from the BSDetector paper (Chen & Mueller, ACL'24). The library provides a simple API to evaluate the trustworthiness of any LLM-generated answer.

## Time Breakdown

**Total time: ~2 hours**

- **Research & Understanding (30 min)**
  - Read Section 3.2 of the paper carefully
  - Analyzed the prompt templates in Figure 6b
  - Tested chat.cleanlab.ai to understand expected behavior
  - Reviewed litellm documentation for model integration

- **Design & Implementation (45 min)**
  - Designed clean API with single main method
  - Implemented core `TrustworthinessDetector` class
  - Added response parsing with regex
  - Implemented caching for efficiency
  - Created batch evaluation feature

- **Testing & Examples (30 min)**
  - Created comprehensive usage examples
  - Tested with assignment examples
  - Added test suite with mocking
  - Verified high/low score behavior

- **Documentation & Polish (15 min)**
  - Wrote clear README with examples
  - Added inline documentation
  - Created configuration system
  - Final code cleanup

## Tools and Resources Used

### Development Tools
- **VS Code**: Primary IDE with Python extension
- **GitHub Copilot**: Assisted with boilerplate code and docstrings
- **Claude AI**: Helped understand the paper and design patterns
- **Git/GitHub**: Version control and hosting

### Python Libraries
- **litellm**: Universal LLM client (recommended by assignment)
- **python-dotenv**: Environment variable management
- **pytest**: Testing framework
- **regex**: Built-in for response parsing

### External Resources
- BSDetector paper (arxiv:2308.16175) - Primary reference
- litellm documentation - For API integration
- Stack Overflow - For regex patterns
- Cleanlab chat tool - To understand expected behavior

### API Services
- **Google Gemini API**: Used as recommended (free tier)
- Tested implementation with actual API calls

## Key Design Decisions

### 1. Simplified Scope
Following instructions, I implemented only self-reflection certainty (Section 3.2), not the full BSDetector algorithm. This kept the implementation focused and manageable within the time constraint.

### 2. Clean API Design
Created a simple interface with one main method: `get_trustworthiness_score(question, answer) -> float`. This makes the library easy to use while hiding complexity.

### 3. Centralized Configuration
Implemented a configuration system where users set `DEFAULT_MODEL` in `.env` once, and it's used throughout the library. This improves usability and reduces errors.

### 4. Robust Parsing
Used flexible regex patterns to handle various LLM response formats. Defaults to 0.5 (uncertain) when parsing fails, ensuring the system degrades gracefully.

### 5. Performance Optimization
Added optional response caching to avoid redundant API calls during development and testing. This significantly reduces costs and latency.

## Challenges and Solutions

### Challenge 1: Response Format Variations
**Problem**: LLMs don't always follow the exact output format.
**Solution**: Implemented flexible regex that handles variations like "answer: A", "answer: [A]", "answer: (A)".

### Challenge 2: Model Selection
**Problem**: Users might have different API keys (OpenAI, Gemini, etc.).
**Solution**: Created a configuration system that automatically validates API keys and uses appropriate models.

### Challenge 3: Testing Without API Calls
**Problem**: Need to test without making expensive API calls.
**Solution**: Used unittest.mock to simulate LLM responses in tests.

## Results

The implementation successfully:
- ✅ Gives high scores (>0.8) to correct answers
- ✅ Gives low scores (<0.2) to incorrect answers
- ✅ Gives medium scores (0.4-0.6) to uncertain cases
- ✅ Works with any LLM supported by litellm
- ✅ Handles errors gracefully

## What I Would Do With More Time

1. **Add async support** for parallel API calls
2. **Implement more sophisticated prompting** strategies
3. **Add a web interface** for easy demonstration
4. **Create more comprehensive test suite** with edge cases
5. **Add detailed logging** for debugging
6. **Implement the full BSDetector** algorithm for comparison

## Conclusion

This implementation provides a clean, usable solution to the assignment that correctly identifies trustworthy vs untrustworthy LLM outputs. The focus on simplicity and good software engineering practices makes it easy to understand, test, and extend.