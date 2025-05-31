# Python Project

This is a comprehensive Python project template with CI/CD integration, Docker support, and best practices.

## Project Structure

```
.
├── .github/
│   ├── workflows/       # GitHub Actions workflows
│   └── copilot-instructions.md  # Guidelines for GitHub Copilot
├── .vscode/             # VS Code configuration
├── src/                 # Source code
│   ├── __init__.py      # Package initialization
│   ├── main.py          # Main application entry point
│   ├── cli.py           # Command-line interface
│   ├── config.py        # Configuration handling
│   ├── logger.py        # Logging setup
│   └── utils.py         # Utility functions
├── tests/               # Test files
├── config.json          # Application configuration
├── Dockerfile           # For containerization
├── docker-compose.yml   # For local development with Docker
├── Jenkinsfile          # CI/CD pipeline for Jenkins
├── Makefile             # Common development tasks
├── requirements.txt     # Project dependencies
├── setup.py             # Package setup script
└── tox.ini              # Tox configuration for testing
```

## Getting Started

### Local Development

1. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   make dev-install
   # or alternatively:
   pip install -r requirements.txt
   pip install -e .
   ```

3. Run the application:
   ```bash
   make run
   # or alternatively:
   python -m src.main
   ```

4. Format and lint code:
   ```bash
   make format
   make lint
   ```

### Using Docker

1. Build the Docker image:
   ```bash
   make docker-build
   # or alternatively:
   docker-compose build
   ```

2. Run with Docker Compose:
   ```bash
   make docker-run
   # or alternatively:
   docker-compose up
   ```

## Testing

Run tests with coverage:
```bash
make test
# or alternatively:
pytest tests/ --cov=src --cov-report=term-missing
```

Run tests in multiple Python environments:
```bash
tox
```

## CI/CD

This project includes:
- GitHub Actions workflow for continuous integration
- Jenkinsfile for Jenkins pipeline integration

## Adding New Features

1. Create new modules in the `src` directory
2. Add corresponding tests in the `tests` directory
3. Update `requirements.txt` with any new dependencies
