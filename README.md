# E2E Migration MCP Server

A Model Context Protocol (MCP) server for end-to-end validation tests and AI-powered assistance for cloud migration projects.

## Features

- **Terraform Coverage Validation**: AI-powered comparison of Terraform topologies with structured JSON responses
- **Python Test Coverage Validation**: AI-powered comparison of Python test implementations
- **Test Environment Preparation**: Prepare test execution environments and provide run commands
- **File Analysis Tracking**: Complete visibility into which files are analyzed during validation

## Setup

1. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd e2e-validation-mcp
   ```

2. **Configure environment variables** (IMPORTANT: Do this FIRST):
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your configuration:
   ```bash
   # Required paths
   CLOUDN_PATH=/path/to/your/cloudn
   REGRESSION_PATH=/path/to/your/regression_test
   TFVARS_PATH=/path/to/your/provider_cred.tfvars

   # Required LLM configuration
   LLM_API_KEY=your-api-key-here
   LLM_BASE_URL=https://api.openai.com/v1
   LLM_CHAT_MODEL_NAME=gpt-3.5-turbo
   ```

3. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   The server will start on `http://localhost:8001/sse` by default.

## Available Tools

### `terraform_coverage_validation`
Compares Terraform configurations between regression and CloudN paths.

**Parameters:**
- `test_path_under_regression`: Path to expected topology under regression directory
- `test_path_under_cloudn`: Path to actual topology under CloudN directory

**Returns:** JSON with validation result, analysis, file paths, and list of files analyzed.

### `python_coverage_validation`
Compares Python test logic between regression and CloudN paths.

**Parameters:**
- `test_path_under_regression`: Path to expected test logic under regression directory
- `test_path_under_cloudn`: Path to actual test logic under CloudN directory

**Returns:** JSON with validation result ("True"/"Ok"/"False"/"Error"), analysis, file paths, and list of files analyzed.

### `entire_test_validation_preparation`
Prepares the test execution environment and provides run commands.

**Parameters:**
- `test_suite_name`: Name of the test suite to prepare

### `prepare_terraform_syntax_validation`
Prepares the Terraform validation environment for syntax checking.

**Parameters:**
- `test_suite_name`: Name of the test suite to prepare

## JSON Response Format

Both validation tools return structured JSON:

```json
{
  "validation_result": "True|False|Error" (or "Ok" for Python),
  "analysis": "Detailed LLM analysis",
  "expected_*_path": "Full path to expected files",
  "expected_*_files": ["list", "of", "files", "read"],
  "actual_*_path": "Full path to actual files",
  "actual_*_files": ["list", "of", "files", "read"],
  "error": "Error message or null"
}
```

## MCP Client Configuration

Add this to your MCP client:

```json
{
  "mcpServers": {
    "e2e-validation-agent": {
      "name": "e2e-validation-agent",
      "url": "http://localhost:8001/sse",
      "enabled": true
    }
  }
}
```

## Troubleshooting

- **Environment variables not set**: Ensure all required variables in `.env` are configured
- **Port in use**: Change `PORT` in `.env` to use a different port
- **Permission denied**: Run `chmod +x setup.sh` before executing the setup script
