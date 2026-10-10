# Contributing to LLMs Course

Thanks for your interest in contributing! This guide will help you get started.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Include: what you expected, what happened, steps to reproduce
- Mention your Python version and OS

### Suggesting Improvements

- Open an issue describing the improvement
- Include why it would be helpful
- Reference any related resources

### Submitting Code

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test your code (see below)
5. Submit a pull request

## Code Standards

### Python

- Follow PEP 8 style
- Use type hints where practical
- Include docstrings for functions and classes
- Keep files under 500 lines (split if longer)
- Use meaningful variable names

### README Files

- Use clear, concise language
- Include code examples
- Link to related modules
- Keep formatting consistent with existing READMEs

### Dependencies

- Minimize new dependencies
- Add new packages to `requirements.txt`
- Pin versions when possible

## Testing

Before submitting a PR:

```bash
# Check Python syntax
python -m py_compile your_file.py

# Test your code runs without errors
python your_file.py --help

# If you added an API call, verify it works with a real key
```

## File Structure

```
LLMs/
├── 01-introduction/        # Module 1: Introduction
├── 02-transformer-architecture/  # Module 2: Transformers
├── 02-prompt-engineering/  # Module 3: Prompt Engineering
├── 03-llm-apis/            # Module 4: LLM APIs
├── 04-rag/                 # Module 5: RAG
├── 05-fine-tuning/         # Module 6: Fine-Tuning
├── 07-advanced-prompting/  # Module 7: Advanced Prompting
├── 08-ai-agents/           # Module 8: AI Agents
├── 09-advanced-rag/        # Module 9: Advanced RAG
├── 10-evaluation/          # Module 10: Evaluation
├── 11-deployment/          # Module 11: Deployment
├── 12-safety-ethics/       # Module 12: Safety & Ethics
└── projects/               # Practice projects
```

## Getting Help

- Open an issue for questions
- Check existing documentation first
- Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
