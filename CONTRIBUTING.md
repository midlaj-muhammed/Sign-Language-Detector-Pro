# Contributing to Sign Language Detector Pro

Thank you for your interest in contributing to Sign Language Detector Pro! This document provides guidelines and information for contributors.

## 🌟 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new functionality
- **Code Contributions**: Implement new features or fix bugs
- **Documentation**: Improve guides, examples, and API docs
- **Testing**: Add test cases and improve coverage
- **ASL Expertise**: Help improve gesture recognition accuracy

## 🚀 Getting Started

### Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/midlaj-muhammed/Sign-Language-Detector-Pro.git
   cd Sign-Language-Detector-Pro
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

5. **Run Tests**
   ```bash
   python -m pytest tests/
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   python -m pytest

   # Run specific test files
   python test_gemini_api.py
   python test_fallback_classifier.py
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new ASL gesture recognition for letter X"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide

- Follow **PEP 8** Python style guidelines
- Use **type hints** for function parameters and return values
- Write **docstrings** for all functions, classes, and modules
- Keep line length under **88 characters** (Black formatter standard)

### Code Quality Tools

We use the following tools to maintain code quality:

```bash
# Code formatting
black src/ tests/

# Import sorting
isort src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Example Code Style

```python
from typing import Dict, List, Optional, Tuple

def classify_gesture(
    gesture_description: str,
    confidence_threshold: float = 0.5,
    use_fallback: bool = True
) -> Dict[str, Any]:
    """
    Classify a sign language gesture based on description.
    
    Args:
        gesture_description: Detailed description of hand gesture
        confidence_threshold: Minimum confidence for valid prediction
        use_fallback: Whether to use fallback classifier if AI fails
        
    Returns:
        Dictionary containing prediction results with confidence scores
        
    Raises:
        ValueError: If gesture_description is empty or invalid
    """
    if not gesture_description.strip():
        raise ValueError("Gesture description cannot be empty")
    
    # Implementation here
    return {
        "prediction": "A",
        "confidence": 0.95,
        "method": "gemini_ai"
    }
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows

### Writing Tests

```python
import pytest
from src.fallback_classifier import FallbackSignLanguageClassifier

class TestFallbackClassifier:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.classifier = FallbackSignLanguageClassifier()
    
    def test_classify_pointing_gesture(self):
        """Test classification of pointing gesture."""
        description = "Hand: Right; Extended fingers: index; Closed fingers: thumb, middle, ring, pinky"
        result = self.classifier.classify_gesture(description)
        
        assert result["success"] is True
        assert result["letter"] == "1"
        assert result["confidence"] > 0.7
    
    def test_invalid_input_handling(self):
        """Test handling of invalid input."""
        with pytest.raises(ValueError):
            self.classifier.classify_gesture("")
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_fallback_classifier.py

# Run tests with verbose output
pytest -v
```

## 📚 Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def process_image(image_path: str, detect_hands: bool = True) -> Dict[str, Any]:
    """
    Process a sign language image and extract gesture information.
    
    This function loads an image, detects hands using MediaPipe, and
    extracts detailed gesture descriptions for classification.
    
    Args:
        image_path: Path to the image file to process
        detect_hands: Whether to perform hand detection
        
    Returns:
        Dictionary containing:
            - hands_detected: Number of hands found
            - gesture_descriptions: List of gesture descriptions
            - processing_time: Time taken in seconds
            
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image format is not supported
        
    Example:
        >>> result = process_image("examples/letter_a.jpg")
        >>> print(result["hands_detected"])
        1
    """
```

### README Updates

When adding new features:

1. Update the feature list in README.md
2. Add usage examples
3. Update performance metrics if applicable
4. Add any new dependencies to installation instructions

## 🎯 Contribution Areas

### High Priority

1. **ASL Gesture Expansion**
   - Add support for new letters, numbers, or words
   - Improve recognition accuracy for existing gestures
   - Add support for regional ASL variations

2. **Performance Optimization**
   - Reduce processing time for large files
   - Optimize memory usage
   - Improve hand detection accuracy

3. **User Experience**
   - Enhance Streamlit interface
   - Add new visualization options
   - Improve error messages and user feedback

### Medium Priority

1. **Testing and Quality**
   - Increase test coverage
   - Add performance benchmarks
   - Create automated testing workflows

2. **Documentation**
   - Add video tutorials
   - Create API documentation
   - Write deployment guides

### Feature Requests

Before implementing new features:

1. **Check existing issues** to avoid duplication
2. **Create an issue** to discuss the feature
3. **Get feedback** from maintainers
4. **Plan the implementation** with clear requirements

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** for similar problems
2. **Try the latest version** to see if it's already fixed
3. **Check documentation** for known limitations

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. Upload file '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 10, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- Browser: [e.g., Chrome 96.0]
- Sign Language Detector Pro Version: [e.g., 1.0.0]

**Additional Context**
- Error messages or logs
- Screenshots if applicable
- Sample files that cause the issue
```

## 📋 Pull Request Process

### Before Submitting

1. **Ensure tests pass**: All existing tests should continue to pass
2. **Add new tests**: For any new functionality
3. **Update documentation**: Including docstrings and README if needed
4. **Follow coding standards**: Use the tools mentioned above

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added and passing
```

### Review Process

1. **Automated checks** must pass (tests, linting, etc.)
2. **Code review** by at least one maintainer
3. **Testing** on different environments if needed
4. **Approval** and merge by maintainer

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** contributors section
- **CHANGELOG.md** for significant contributions
- **GitHub releases** for version contributions

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [maintainer-email] for private inquiries

## 📄 License

By contributing to Sign Language Detector Pro, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Sign Language Detector Pro! Your efforts help make sign language recognition more accessible to everyone. 🤟
