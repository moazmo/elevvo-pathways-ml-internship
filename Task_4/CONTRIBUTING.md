# Contributing to Walmart Sales Forecasting System

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- Node.js 16+
- Git

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/elevvo-pathways-ml-internship.git
   cd elevvo-pathways-ml-internship/Task_4
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```
4. Start development environment:
   ```bash
   python start_full_system.py --frontend
   ```

## 🛠️ Development Guidelines

### Code Style
- **Python**: Follow PEP 8 guidelines
- **JavaScript/React**: Use ESLint and Prettier configurations
- **SQL**: Use consistent formatting and naming conventions

### Commit Messages
Use conventional commit format:
```
type(scope): description

Examples:
feat(api): add batch prediction endpoint
fix(frontend): resolve prediction form validation
docs(readme): update installation instructions
```

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

## 🧪 Testing

### Backend Tests
```bash
python -m pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend && npm test
```

### Integration Tests
```bash
python start_full_system.py --frontend
# Test all endpoints and UI functionality
```

## 📝 Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the guidelines above
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Ensure all tests pass**
6. **Submit a pull request** with:
   - Clear description of changes
   - Screenshots for UI changes
   - Test results
   - Any breaking changes noted

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots for UI changes
```

## 🐛 Bug Reports

When reporting bugs, please include:
- **Environment details** (OS, Python version, etc.)
- **Steps to reproduce** the issue
- **Expected behavior**
- **Actual behavior**
- **Error messages** or logs
- **Screenshots** if applicable

## 💡 Feature Requests

For new features, please provide:
- **Use case** description
- **Proposed solution**
- **Alternative solutions** considered
- **Additional context**

## 📚 Documentation

### Code Documentation
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Comment complex logic

### API Documentation
- Update OpenAPI/Swagger documentation
- Include request/response examples
- Document error codes and messages

## 🏗️ Architecture Guidelines

### Backend (FastAPI)
- Use dependency injection for database sessions
- Implement proper error handling
- Follow RESTful API conventions
- Add logging for important operations

### Frontend (React)
- Use functional components with hooks
- Implement proper error boundaries
- Follow component composition patterns
- Maintain responsive design

### Database
- Use migrations for schema changes
- Follow naming conventions
- Add proper indexes for performance
- Document complex queries

## 🔒 Security

- Never commit sensitive information (API keys, passwords)
- Use environment variables for configuration
- Validate all user inputs
- Follow OWASP security guidelines

## 📊 Performance

- Monitor API response times
- Optimize database queries
- Use caching where appropriate
- Profile code for bottlenecks

## 🤝 Community

- Be respectful and inclusive
- Help other contributors
- Share knowledge and best practices
- Follow the code of conduct

## 📞 Getting Help

- Create an issue for bugs or questions
- Join discussions in existing issues
- Check documentation first
- Provide detailed information when asking for help

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Project documentation

Thank you for contributing to make this project better! 🚀