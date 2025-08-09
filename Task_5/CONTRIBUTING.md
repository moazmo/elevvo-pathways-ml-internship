# Contributing to Traffic Sign Recognition

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/elevvo-pathways-ml-internship.git
   cd elevvo-pathways-ml-internship/Task_5
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Download the dataset** (see data/README.md)

5. **Run tests**
   ```bash
   python test_app.py
   ```

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use Black for code formatting: `black src/ webapp/`
- Use meaningful variable and function names
- Add docstrings to all functions and classes

### Testing
- Write tests for new features
- Ensure all existing tests pass
- Test both happy path and edge cases
- Include integration tests for API endpoints

### Documentation
- Update README.md for new features
- Add docstrings to new functions
- Update API documentation if endpoints change
- Include examples in docstrings

## 🔄 Contribution Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   python test_app.py
   black --check src/ webapp/
   flake8 src/ webapp/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces
- Screenshots if applicable

## 💡 Feature Requests

For new features, please:
- Check if the feature already exists
- Describe the use case and benefits
- Provide implementation suggestions if possible
- Consider backward compatibility

## 📋 Project Structure

```
├── src/                 # Core ML modules
├── webapp/             # Flask web application
├── notebooks/          # Jupyter notebooks for ML pipeline
├── data/              # Dataset (not in repo)
├── models/            # Trained models (not in repo)
├── production/        # Production model (not in repo)
├── tests/             # Test files
└── docs/              # Documentation
```

## 🏷️ Commit Message Format

Use conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

## 📞 Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.