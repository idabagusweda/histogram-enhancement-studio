# Contributing to Histogram Enhancement Studio

Contributions and feedback are welcome! To contribute to this project, please follow the guidelines below.

## Getting Started

1. Fork the repository on GitHub: `https://github.com/idabagusweda/historgram-enhancement-studio`.
2. Clone your forked repository locally:
   ```bash
   git clone https://github.com/<your-username>/historgram-enhancement-studio.git
   cd historgram-enhancement-studio
   ```
3. Set up a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Run the Streamlit development server:
   ```bash
   streamlit run app.py
   ```

## Development and Code Style

- **Python Version:** Python 3.10 or higher.
- **Modularity:** Keep processing logic separate from UI orchestration. Add or update algorithm implementations inside `modules/`, maintaining clean type annotations and docstrings.
- **Dependencies:** Avoid introducing heavy dependencies when standard libraries or existing scientific packages (`numpy`, `scipy`, `scikit-image`, `opencv-python-headless`) can handle the task.
- **Formatting:** Adhere to PEP 8 standards. Keep functions focused and deterministic.

## Submitting Pull Requests

1. Create a feature branch with a descriptive name:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Test your changes locally to ensure no regressions in image processing or chart rendering.
3. Commit your changes with clear, concise commit messages.
4. Push your branch to your fork and submit a Pull Request to the main branch.
5. Provide a summary of changes, problem context, and before/after screenshots if UI elements or visualizers were modified.
