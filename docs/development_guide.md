# QTrobot Confusion Detection System - Development Guide

This guide provides information for developers who want to extend or modify the QTrobot Confusion Detection System.

## Table of Contents

1. [Development Environment](#development-environment)
2. [Project Structure](#project-structure)
3. [Adding New Features](#adding-new-features)
4. [Modifying Existing Components](#modifying-existing-components)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Performance Optimization](#performance-optimization)
8. [Deployment](#deployment)

## Development Environment

### Prerequisites

- Ubuntu 20.04 LTS
- ROS Noetic
- Python 3.8+
- Git
- Docker (optional, for containerized development)

### Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/confusion_project.git
   cd confusion_project
   ```

2. Run the setup script:
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. Build the ROS packages:
   ```bash
   cd src
   catkin_make
   ```

4. Source the workspace:
   ```bash
   source devel/setup.bash
   ```

### Development Tools

- **IDE**: We recommend using Visual Studio Code with the following extensions:
  - Python
  - ROS
  - Docker
  - YAML
  - Markdown

- **Code Formatting**: Use `black` for Python code formatting:
  ```bash
  pip install black
  black src/
  ```

- **Linting**: Use `flake8` for Python linting:
  ```bash
  pip install flake8
  flake8 src/
  ```

## Project Structure

The project is organized as follows:

```
confusion_project/
├── config/                  # Configuration files
│   ├── detection_params.yaml
│   └── repair_policies.yaml
├── docker/                  # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── docs/                    # Documentation
│   ├── api_reference.md
│   ├── development_guide.md
│   ├── system_architecture.md
│   └── user_guide.md
├── launch/                  # ROS launch files
│   └── confusion_system.launch
├── models/                  # Pre-trained models
│   ├── confusion_detection/
│   └── face_detection/
├── scripts/                 # Utility scripts
│   ├── docker_run.sh
│   ├── evaluation.py
│   ├── run_demo.sh
│   └── setup.sh
├── src/                     # Source code
│   ├── common/              # Common utilities
│   │   ├── src/
│   │   │   ├── utils.py
│   │   │   └── visualization.py
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   ├── confusion_detection/ # Confusion detection package
│   │   ├── launch/
│   │   ├── msg/
│   │   ├── src/
│   │   │   ├── face_detector_node.py
│   │   │   ├── feature_extractor_node.py
│   │   │   └── confusion_classifier_node.py
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   └── repair_policy/       # Repair policy package
│       ├── launch/
│       ├── srv/
│       ├── src/
│       │   ├── policy_engine_node.py
│       │   └── repair_strategies.py
│       ├── CMakeLists.txt
│       └── package.xml
└── test/                    # Tests
    ├── integration/
    └── unit/
```

## Adding New Features

### Adding a New ROS Node

1. Create a new Python file in the appropriate package:
   ```bash
   touch src/confusion_detection/src/new_node.py
   chmod +x src/confusion_detection/src/new_node.py
   ```

2. Implement the node using the ROS Python API:
   ```python
   #!/usr/bin/env python3
   
   import rospy
   
   def main():
       rospy.init_node('new_node')
       # Node implementation
       rospy.spin()
   
   if __name__ == '__main__':
       try:
           main()
       except rospy.ROSInterruptException:
           pass
   ```

3. Add the node to the package's CMakeLists.txt:
   ```cmake
   catkin_install_python(PROGRAMS
     src/new_node.py
     DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
   )
   ```

4. Update the launch file to include the new node:
   ```xml
   <node name="new_node" pkg="confusion_detection" type="new_node.py" output="screen">
     <!-- Parameters -->
   </node>
   ```

### Adding a New Message or Service

1. Create a new message or service definition in the appropriate package:
   ```bash
   touch src/confusion_detection/msg/NewMessage.msg
   # or
   touch src/repair_policy/srv/NewService.srv
   ```

2. Define the message or service:
   ```
   # NewMessage.msg
   Header header
   string data
   float32 value
   ```

3. Update the package's CMakeLists.txt:
   ```cmake
   add_message_files(
     FILES
     NewMessage.msg
   )
   
   generate_messages(
     DEPENDENCIES
     std_msgs
   )
   ```

4. Update the package's package.xml to include message generation and runtime dependencies.

### Adding a New Repair Strategy

1. Open `src/repair_policy/src/repair_strategies.py`

2. Add a new strategy class:
   ```python
   class NewStrategy(RepairStrategy):
       """Description of the new strategy."""
       
       def __init__(self):
           """Initialize the new strategy."""
           super().__init__(
               name="new_strategy",
               description="Description of the new strategy",
               priority=0.5
           )
       
       def is_applicable(self, confusion_state, context):
           """Check if the strategy is applicable."""
           # Implementation
           return True
       
       def generate_repair_message(self, confusion_state, context):
           """Generate a repair message."""
           # Implementation
           return "Repair message"
   ```

3. Add the new strategy to the `REPAIR_STRATEGIES` dictionary:
   ```python
   REPAIR_STRATEGIES = {
       # Existing strategies
       RepairStrategyType.NEW_STRATEGY: NewStrategy(),
   }
   ```

4. Update the configuration file `config/repair_policies.yaml` to include parameters for the new strategy.

## Modifying Existing Components

### Modifying Face Detection

The face detection component is implemented in `src/confusion_detection/src/face_detector_node.py`. To modify it:

1. Update the `load_face_detection_model` method to use a different face detection model.
2. Update the `detect_faces` method to implement a different face detection algorithm.
3. Update the `extract_facial_features` method to extract additional or different features.

### Modifying Confusion Classification

The confusion classification component is implemented in `src/confusion_detection/src/confusion_classifier_node.py`. To modify it:

1. Update the `load_confusion_model` method to use a different confusion classification model.
2. Update the `extract_confusion_features` method to extract additional or different features.
3. Update the `classify_confusion` method to implement a different classification algorithm.

### Modifying Repair Strategies

The repair strategies are implemented in `src/repair_policy/src/repair_strategies.py`. To modify them:

1. Update the existing strategy classes to change their behavior.
2. Update the `select_strategy` function to implement a different strategy selection algorithm.

## Testing

### Unit Testing

Unit tests are located in the `test/unit` directory. To run them:

```bash
cd test/unit
python -m unittest discover
```

To add a new unit test:

1. Create a new Python file in the `test/unit` directory:
   ```bash
   touch test/unit/test_new_feature.py
   ```

2. Implement the test using the `unittest` framework:
   ```python
   import unittest
   
   class TestNewFeature(unittest.TestCase):
       def test_something(self):
           # Test implementation
           self.assertTrue(True)
   
   if __name__ == '__main__':
       unittest.main()
   ```

### Integration Testing

Integration tests are located in the `test/integration` directory. To run them:

```bash
cd test/integration
python -m unittest discover
```

To add a new integration test:

1. Create a new Python file in the `test/integration` directory:
   ```bash
   touch test/integration/test_new_integration.py
   ```

2. Implement the test using the `unittest` framework and ROS testing utilities.

### System Testing

System tests involve running the entire system and evaluating its performance. Use the evaluation script:

```bash
python3 scripts/evaluation.py --mode live --duration 60 --visualize
```

## Documentation

### Code Documentation

Follow these guidelines for code documentation:

- Use docstrings for all functions, classes, and methods.
- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings.
- Include examples where appropriate.
- Document all parameters, return values, and exceptions.

Example:

```python
def function_name(param1, param2):
    """Brief description of the function.
    
    More detailed description of the function.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
    
    Returns:
        Description of the return value.
    
    Raises:
        ExceptionType: Description of when this exception is raised.
    """
    # Implementation
    return result
```

### System Documentation

System documentation is located in the `docs` directory. To update it:

1. Edit the appropriate Markdown file.
2. If adding a new document, update the README.md to include a link to it.

## Performance Optimization

### Model Optimization

To optimize the machine learning models:

1. Convert models to ONNX format for faster inference:
   ```python
   import torch
   import torch.onnx
   
   # Load PyTorch model
   model = torch.load('model.pth')
   model.eval()
   
   # Export to ONNX
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy_input, 'model.onnx')
   ```

2. Use quantization to reduce model size:
   ```python
   import onnx
   from onnxruntime.quantization import quantize_dynamic
   
   # Load ONNX model
   model_path = 'model.onnx'
   quantized_model_path = 'model_quantized.onnx'
   
   # Quantize model
   quantize_dynamic(model_path, quantized_model_path)
   ```

### ROS Node Optimization

To optimize ROS nodes:

1. Use appropriate message queue sizes to avoid memory issues.
2. Use appropriate publishing frequencies to avoid overloading the system.
3. Use efficient data structures and algorithms.
4. Consider using C++ for performance-critical components.

## Deployment

### Native Deployment

To deploy the system on the QTrobot platform:

1. Clone the repository on the QTrobot's NUC computer.
2. Run the setup script and build the ROS packages.
3. Launch the system using the provided launch file.

### Docker Deployment

To deploy the system using Docker:

1. Build the Docker image:
   ```bash
   ./scripts/docker_run.sh --build
   ```

2. Run the Docker container:
   ```bash
   ./scripts/docker_run.sh
   ```

### Continuous Integration/Deployment

The project includes a GitHub Actions workflow for CI/CD. To use it:

1. Fork the repository on GitHub.
2. Enable GitHub Actions for your fork.
3. Push changes to your fork to trigger the workflow.

The workflow will:
- Build the ROS packages
- Run the tests
- Build the Docker image
- Deploy the system (if configured)
