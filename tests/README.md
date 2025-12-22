# Health Check System - Modular Architecture

A professional, modular health check system for the RAG pipeline that validates all critical components before execution.

## 📁 Structure

```
tests/
├── health/
│   ├── __init__.py          # Package exports
│   ├── base.py              # Base classes and result objects
│   ├── runner.py            # Main orchestrator
│   ├── config.py            # Configuration settings
│   ├── utils.py             # Utility functions
│   ├── infrastructure.py    # Infrastructure checks (Ollama, Network)
│   ├── models.py            # Model availability checks
│   ├── components.py        # Component functionality checks
│   └── external.py          # External service checks (Wikipedia)
├── health_checks.py         # Main entry point
└── README.md                # This file
```

## 🏗️ Architecture

### Base Layer (`base.py`)

**HealthCheckResult**
- Encapsulates check results with metadata
- Stores: name, status, message, duration, timestamp
- Provides formatted string representation

**BaseHealthCheck** (Abstract)
- Base class for all check modules
- Enforces `run_checks()` and `category_name` implementation
- Manages result collection and display

### Check Modules

Each module inherits from `BaseHealthCheck` and focuses on a specific domain:

#### 1. Infrastructure Checks (`infrastructure.py`)
- ✅ Ollama service availability
- ✅ Network connectivity

#### 2. Model Checks (`models.py`)
- ✅ Embedding model availability
- ✅ LLM model availability

#### 3. Component Checks (`components.py`)
- ✅ Embedding generation functionality
- ✅ LLM generation functionality
- ✅ Vector store (FAISS) initialization
- ✅ Document processing and chunking
- ✅ Session management

#### 4. External Service Checks (`external.py`)
- ✅ Wikipedia API accessibility

### Runner (`runner.py`)

**HealthCheckRunner**
- Orchestrates all check modules
- Manages check dependencies
- Provides comprehensive reporting
- Returns success/failure status

### Configuration (`config.py`)

**HealthCheckConfig**
- Centralized configuration for all checks
- Customizable timeouts and endpoints
- Defines critical checks

### Utilities (`utils.py`)

Helper functions:
- `timed_check`: Decorator for timing checks
- `safe_import`: Safe module importing
- `format_duration`: Human-readable time formatting
- `truncate_string`: String truncation
- Display utilities for formatted output

## 🚀 Usage

### Basic Usage

```python
from src.config import get_settings
from tests.health import HealthCheckRunner

settings = get_settings()
runner = HealthCheckRunner(settings)
success = runner.run_all_checks(verbose=True)

if success:
    print("All critical checks passed!")
else:
    print("Some checks failed:", runner.get_failed_checks())
```

### Command Line

```bash
# Run all health checks
python tests/health_checks.py

# Or use the convenience script
python test_system.py

# Or use the batch/shell scripts
run_with_checks.bat    # Windows
./run_with_checks.sh   # Linux/Mac
```

### Programmatic Access

```python
from tests.health import (
    HealthCheckRunner,
    InfrastructureChecks,
    ModelChecks,
    ComponentChecks,
    ExternalServiceChecks
)

# Run specific check modules
settings = get_settings()

infra = InfrastructureChecks(settings)
results = infra.run_checks()

for result in results:
    print(f"{result.name}: {'PASS' if result.passed else 'FAIL'}")
```

## 🔧 Extending the System

### Adding a New Check Module

1. Create a new file in `tests/health/`
2. Inherit from `BaseHealthCheck`
3. Implement required methods:

```python
from .base import BaseHealthCheck, HealthCheckResult

class MyCustomChecks(BaseHealthCheck):
    
    @property
    def category_name(self) -> str:
        return "My Custom Checks"
    
    def run_checks(self) -> list[HealthCheckResult]:
        self.results = []
        self.check_something()
        return self.results
    
    def check_something(self) -> HealthCheckResult:
        try:
            # Your check logic here
            success = True
            return self.add_result(
                "My Check",
                success,
                "Check passed",
                metadata={"key": "value"}
            )
        except Exception as e:
            return self.add_result(
                "My Check",
                False,
                f"Error: {str(e)}"
            )
```

4. Register in `runner.py`:

```python
from .my_custom import MyCustomChecks

# In run_all_checks method:
custom_checks = MyCustomChecks(self.settings)
check_sequence.append((5, 5, custom_checks))
```

### Adding a New Individual Check

Add a method to an existing check module:

```python
def check_new_feature(self) -> HealthCheckResult:
    try:
        # Your check logic
        return self.add_result(
            "New Feature",
            True,
            "Working correctly"
        )
    except Exception as e:
        return self.add_result(
            "New Feature",
            False,
            f"Error: {str(e)}"
        )
```

Then call it in `run_checks()`:

```python
def run_checks(self) -> list[HealthCheckResult]:
    self.results = []
    self.check_existing_feature()
    self.check_new_feature()  # Add this
    return self.results
```

## 📊 Check Dependencies

The runner automatically handles dependencies:

```
Infrastructure Checks (Always run)
    ↓
Model Checks (Requires: Ollama running)
    ↓
Component Checks (Requires: Ollama + Models available)
    ↓
External Checks (Requires: Network connectivity)
```

## 🎯 Critical vs Non-Critical Checks

**Critical Checks** (Must pass for system to run):
- Ollama Service
- Embedding Model
- LLM Model
- Embedding Generation
- LLM Generation

**Non-Critical Checks** (Warnings only):
- Network Connectivity
- Wikipedia API
- Vector Store
- Document Processing
- Session Management

Configure in `config.py`:

```python
CRITICAL_CHECKS = [
    "Ollama Service",
    "Embedding Model",
    "LLM Model",
]
```

## 🧪 Testing

The health check system is self-testing. Run it to verify:

```bash
python tests/health_checks.py
```

## 📈 Performance

Each check includes timing information:
- Individual check duration
- Total execution time
- Metadata for performance analysis

## 🔍 Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

runner = HealthCheckRunner(settings)
runner.run_all_checks(verbose=True)
```

## 📝 Best Practices

1. **Keep checks independent**: Each check should not depend on others
2. **Use timeouts**: All network calls should have timeouts
3. **Handle exceptions**: Wrap checks in try-except blocks
4. **Provide helpful messages**: Include actionable error messages
5. **Add metadata**: Store useful debugging information
6. **Time your checks**: Use the timing utilities

## 🤝 Contributing

When adding new checks:
1. Follow the existing module pattern
2. Add comprehensive error handling
3. Include helpful error messages
4. Update this README
5. Test thoroughly

## 📄 License

Part of the RAG-Scale project.
