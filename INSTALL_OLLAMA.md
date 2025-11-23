# Installing Ollama for LLM-Based Topic Analysis

To use LLM-powered topic analysis for better accuracy, install Ollama:

## Windows Installation

1. **Download Ollama:**
   - Visit: https://ollama.com/download
   - Download the Windows installer
   - Run the installer

2. **Install a Model:**
   ```bash
   ollama pull llama2
   ```
   Or for better performance:
   ```bash
   ollama pull mistral
   ollama pull llama2:13b
   ```

3. **Verify Installation:**
   ```bash
   ollama list
   ```

4. **Start Ollama Service:**
   - Ollama runs as a service on Windows
   - It should start automatically after installation
   - If not, run: `ollama serve`

## Python Package

Install the Python client:
```bash
pip install ollama
```

## Usage

Once installed, the system will automatically use Ollama for:
- Understanding full conversation context
- Extracting main topics/issues accurately
- Creating comprehensive summaries

The system will fall back to enhanced rule-based analysis if Ollama is not available.

