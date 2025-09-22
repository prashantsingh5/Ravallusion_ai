## 🛠️ Installation

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd final_module

```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv moduleenv

# Activate virtual environment
# On Windows:
moduleenv\Scripts\activate
# On macOS/Linux:
source moduleenv/bin/activate
```

### 3. Install Dependencies

The `ravallusion_ai` module handles most dependencies automatically. You only need to install a few additional packages:

```bash
# Install minimal additional dependencies
pip install -r requirements.txt

```

### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env
```

### 5. Verify Installation
```bash
# Test the AI evaluation module
python evaluation_test.py # for testing purpose

# Start the API server
python app.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file according to .env.example:
