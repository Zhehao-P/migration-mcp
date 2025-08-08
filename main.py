"""
This module contains the main entry point for the E2E Validation MCP server.
It defines an MCP server for running specific tests and returning execution output.
"""

import os
import asyncio
import subprocess
import glob
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from asyncio.log import logger
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context
from openai import AsyncOpenAI

load_dotenv()
CLOUDN_PATH: str = os.getenv("CLOUDN_PATH", "")
REGRESSION_PATH: str = os.getenv("REGRESSION_PATH", "")
TFVARS_PATH: str = os.getenv("TFVARS_PATH", "")
TFVARS_FILE_NAME = "provider_cred.tfvars"

# LLM Configuration
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_CHAT_MODEL_NAME: str = os.getenv("LLM_CHAT_MODEL_NAME", "gpt-3.5-turbo")

# Base directory
BASE_DIR = Path(__file__).parent


SYSTEM_PROMPT = """
This MCP server is used to validate the E2E test suite.
It provides resources to help you understand the repository structure and the test framework.
It also provides tools to validate both Terraform topology and python test logic coverage
and syntax.
Please use the tools to validate the migration work you have done, and fix all issues found.

IMPORTANT:
- Always check the context here when you need more context about the migration task.
- Always use both coverage validation tools again when any changes are made to the test suite.
- Always use Terraform syntax validation after both coverage validation results are acceptable.
- Always use Python syntax validation at very end when all other tools hit no issue.
- Always run the syntax validation tools command in the given directory.

If python syntax validation is run, and both report and logs show no error you can claim the
test suite migration is complete.
"""

CLOUDN_E2E_FRAMEWORK_PROMPT = """
In e2e test, the test framework will provide the provider credential file, and apply
the Terraform module to create the test topology before running the test.
You can use the test with name mc_spoke_transit_spoke_test under the test directory
as an example to understand the test framework.
vars.tf will be used to get variables for the test from the provider credential file
provided by the test framework. And outputs.tf will be used to export Terraform output
values for the test.
vm_east = tf.vm(tf.outputs["aws_vm_east_public"]["value"]["name"]) in the
mc_spoke_transit_spoke_test.py is an example of how to get the value of the Terraform
output for the test.
"""


def read_files_with_extension(
    directory_path: Path, file_extension: str
) -> tuple[str, list[str]]:
    """
    Helper function to read all files with a specific extension from a directory.

    Args:
        directory_path: Path to the directory to search
        file_extension: File extension to search for (e.g., "tf", "py")

    Returns:
        Tuple containing:
        - String containing all file contents with proper formatting
        - List of file names that were successfully read
    """
    file_context = ""
    files_read = []
    search_pattern = str(directory_path / f"*.{file_extension}")
    files_found = glob.glob(search_pattern)

    if files_found:
        file_context += f"Files from: {directory_path}\n\n"
        for file_path in sorted(files_found):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                file_name = Path(file_path).name
                file_context += f"=== File: {file_name} ===\n"
                file_context += file_content
                file_context += f"\n=== End of {file_name} ===\n\n"
                files_read.append(file_name)
            except Exception as e:
                file_context += f"Error reading {file_path}: {str(e)}\n\n"
    else:
        file_context = f"No .{file_extension} files found in {directory_path}"

    return file_context, files_read


@dataclass
class MCPContext:
    """Context for managing test execution."""

    test_execution_dir: Path
    log_dir: Path
    openai_client: AsyncOpenAI


async def test_openai_client_ready(client: AsyncOpenAI) -> None:
    """
    Test if the OpenAI client is properly configured and ready to use.

    Args:
        client: The AsyncOpenAI client instance to test

    Raises:
        RuntimeError: If the client is not ready or connection fails
    """
    try:
        logger.info("Testing OpenAI client connectivity...")

        # Make a simple API call to test connectivity
        # Using a minimal completion request to verify the client works
        response = await client.chat.completions.create(
            model=LLM_CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
            timeout=10.0,  # 10 second timeout
        )

        if response and response.choices:
            logger.info("✅ OpenAI client is ready and connected successfully")
        else:
            raise RuntimeError("OpenAI client test failed: Empty response received")

    except Exception as e:
        error_msg = f"OpenAI client test failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e


@asynccontextmanager
async def mcp_lifespan(server: FastMCP) -> AsyncIterator[MCPContext]:
    logger.info("Initialize TestContext")
    test_execution_dir = Path(CLOUDN_PATH) / "test-scripts" / "end-to-end"
    log_dir = test_execution_dir / "logs"

    # Initialize OpenAI client
    openai_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    # Test OpenAI client connectivity
    await test_openai_client_ready(openai_client)

    try:
        yield MCPContext(
            test_execution_dir=test_execution_dir,
            log_dir=log_dir,
            openai_client=openai_client,
        )
    finally:
        logger.info("Cleanup TestContext")
        await openai_client.close()


# Initialize FastMCP server
mcp = FastMCP(
    "e2e-migration-agent",
    host=os.getenv("HOST") or "0.0.0.0",
    port=int(os.getenv("PORT") or "8001"),
    instructions=SYSTEM_PROMPT,
    lifespan=mcp_lifespan,
)


@mcp.tool(name="get_context")
async def get_context() -> str:
    """
    This tool is used to get the context about the migration task.
    It provides important settings and necessary information for the E2E
    test suite migration task and this MCP server.
    """

    response = (
        f"Regression test directory path: {REGRESSION_PATH}\n"
        f"Regression libraries path: {REGRESSION_PATH}/avxt/lib/, {REGRESSION_PATH}/autotest/lib/api_pages/\n"
        f"Regression Terraform module path: {REGRESSION_PATH}/avxt/terraform/\n"
        f"\nCloudn directory path: {CLOUDN_PATH}\n"
        f"Cloudn e2e test directory path: {CLOUDN_PATH}/test-scripts/end-to-end/\n"
        f"Cloudn e2e test libraries path: {CLOUDN_PATH}/test-scripts/end-to-end/avxtflib/\n"
        f"Cloudn e2e test Terraform module path: {CLOUDN_PATH}/test-scripts/end-to-end/vendor/\n"
        f"{CLOUDN_E2E_FRAMEWORK_PROMPT}"
    )
    return response


@mcp.tool(name="terraform_coverage_validation")
async def terraform_coverage_validation(
    ctx: Context,
    test_path_under_regression: str,
    test_path_under_cloudn: str,
) -> str:
    """
    This tool is used to validate the Terraform topology coverage for the input test suite.
    It compares the testbed topology using all .tf files from both regression and cloudn paths
    and evaluate if the migrated e2e test topology is the same as the original regression
    test topology.

    Args:
        test_path_under_regression: A relative path relative to the regression directory to the expected topology
        test_path_under_cloudn: A relative path relative to the cloudn directory to the actual topology

    Returns:
        JSON string containing validation result with structure:
        {
            "validation_result": "True" | "False" | "Error",
            "analysis": "Detailed analysis from LLM",
            "expected_topology_path": "full path to expected topology",
            "actual_topology_path": "full path to actual topology",
            "error": "Error message if validation failed" | null
        }
    """

    system_prompt = """
    You are a helpful Hashicorp Terraform expert.
    You will be given two Terraform topology, one is the expected topology,
    and the other is the actual topology.
    You will need to validate the actual topology against the expected topology.

    Please focus on the following aspects:
    1. The number of resources in the topology
    2. The type of resources in the topology
    3. The configuration of the resources in the topology
    4. The relationship between the resources in the topology

    If they will be creating the exact same topology, you can simply return "True".
    If they will be creating different topology, like some of the resources are missing,
    you will need to return "False".
    You will need to return the reason for the difference. Using the following format:

    '''
    1. Expected topology resource: <expected_topology_resource_context>
    Actual topology resource: <actual_topology_resource_context>
    Reason: These two topology resources are different.
    2. Expected topology resource: <expected_topology_resource_context>
    Actual topology resource: None
    Reason: The actual topology resource is missing.
    3. Expected topology resource: <expected_topology_resource_context>
    Actual topology resource: <actual_topology_resource_context>
    Reason: <reason>
    ...
    '''
    """

    # Build full paths
    regression_full_path = (
        Path(REGRESSION_PATH) / test_path_under_regression / "testbed" / "topology"
    )
    cloudn_full_path = Path(CLOUDN_PATH) / test_path_under_cloudn

    try:
        # Read all .tf files using helper function
        expected_topology_context, expected_files = read_files_with_extension(
            regression_full_path, "tf"
        )
        actual_topology_context, actual_files = read_files_with_extension(
            cloudn_full_path, "tf"
        )

        logger.info("Expected terraform files %s", expected_files)
        logger.info("Actual terraform files %s", actual_files)

        question_prompt = f"""
        Start of the expected topology context:
        {expected_topology_context}
        End of the expected topology context.

        Start of the actual topology context:
        {actual_topology_context}
        End of the actual topology context.

        Please validate if these two Terraform topology are the same.
        """

        openai_client = ctx.request_context.lifespan_context.openai_client

        response = await openai_client.chat.completions.create(
            model=LLM_CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_prompt},
            ],
            temperature=0.3,
        )

        # Prepare the analysis result
        analysis_result = ""
        if response.choices and len(response.choices) > 0:
            analysis_result = (
                response.choices[0].message.content or "No analysis generated"
            )
        else:
            analysis_result = "No response received from LLM"

        # Determine validation result based on analysis
        validation_result = "True"
        if "False" in analysis_result or "different" in analysis_result.lower():
            validation_result = "False"
        elif "error" in analysis_result.lower() or "No response" in analysis_result:
            validation_result = "Error"

        # Create structured response
        result = {
            "validation_result": validation_result,
            "analysis": analysis_result,
            "expected_topology_path": str(regression_full_path),
            "actual_topology_path": str(cloudn_full_path),
            "error": None,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        error_result = {
            "validation_result": "Error",
            "analysis": "",
            "expected_topology_path": str(regression_full_path),
            "actual_topology_path": str(cloudn_full_path),
            "error": (
                f"Error validating Terraform topology: {str(e)}. "
                + "Please check if the test path is correct. "
                + f"Regression path {regression_full_path} "
                + f"or cloudn path {cloudn_full_path} is not correct."
            ),
        }
        return json.dumps(error_result, indent=2)


@mcp.tool(name="python_coverage_validation")
async def python_coverage_validation(
    ctx: Context,
    test_path_under_regression: str,
    test_path_under_cloudn: str,
) -> str:
    """
    This tool is used to compare the Python test logic coverage for the input test suite.
    It compares the test logic using all .py files from both regression and cloudn paths
    and evaluate if the migrated e2e test logic is the same as the original regression
    test logic.

    Args:
        test_path_under_regression: A relative path relative to the regression directory to the expected test logic
        test_path_under_cloudn: A relative path relative to the cloudn directory to the actual test logic

    Returns:
        JSON string containing validation result with structure:
        {
            "validation_result": "True" | "Ok" | "False" | "Error",
            "analysis": "Detailed analysis from LLM",
            "expected_test_logic_path": "full path to expected test logic",
            "actual_test_logic_path": "full path to actual test logic",
            "error": "Error message if validation failed" | null
        }
    """

    system_prompt = """
    You are a helpful Python testing expert with expertise in network testing.
    You will be given two Python test implementations, one is the expected test logic,
    and the other is the actual test logic.
    You will need to validate the actual test logic against the expected test logic.

    Focus on comparing:
    1. The total number of test cases
    2. Test structure and flow
    3. Test assertions and validations
    4. Test setup and teardown logic
    5. Key test scenarios and edge cases
    6. Test data and configurations

    If they have exact same or equivalent test logic, you can simply return "True".
    If they have similar test logic, but some implementation details are different,
    you will need to return "Ok".
    If they have different test logic, like some test cases are missing,
    you will need to return "False".
    You will need to return the reason for the difference for "False" or "Ok" cases
    using the following format:

    '''
    1. Expected test logic: <expected_test_logic_context>
    Actual test logic: <actual_test_logic_context>
    Reason: These two test logic implementations are different.
    2. Expected test logic: <expected_test_logic_context>
    Actual test logic: None
    Reason: The actual test logic is missing this test case.
    3. Expected test logic: <expected_test_logic_context>
    Actual test logic: <actual_test_logic_context>
    Reason: <reason>
    ...
    '''
    """
    # Build full paths
    regression_full_path = Path(REGRESSION_PATH) / test_path_under_regression
    cloudn_full_path = Path(CLOUDN_PATH) / test_path_under_cloudn

    try:

        # Read all .py files using helper function
        expected_test_context, expected_files = read_files_with_extension(
            regression_full_path, "py"
        )
        actual_test_context, actual_files = read_files_with_extension(
            cloudn_full_path, "py"
        )

        logger.info("Expected python files %s", expected_files)
        logger.info("Actual python files %s", actual_files)

        question_prompt = f"""
        Start of the expected test logic context:
        {expected_test_context}
        End of the expected test logic context.

        Start of the actual test logic context:
        {actual_test_context}
        End of the actual test logic context.

        Please validate if these two Python test implementations have the same test logic.
        Focus on test structure, assertions, test cases, and overall testing approach.
        """

        openai_client = ctx.request_context.lifespan_context.openai_client

        response = await openai_client.chat.completions.create(
            model=LLM_CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_prompt},
            ],
            temperature=0.3,
        )

        # Prepare the analysis result
        analysis_result = ""
        if response.choices and len(response.choices) > 0:
            analysis_result = (
                response.choices[0].message.content or "No analysis generated"
            )
        else:
            analysis_result = "No response received from LLM"

        # Determine validation result based on analysis
        validation_result = "True"
        if "False" in analysis_result:
            validation_result = "False"
        elif "Ok" in analysis_result or "similar" in analysis_result.lower():
            validation_result = "Ok"
        elif "error" in analysis_result.lower() or "No response" in analysis_result:
            validation_result = "Error"

        # Create structured response
        result = {
            "validation_result": validation_result,
            "analysis": analysis_result,
            "expected_test_logic_path": str(regression_full_path),
            "actual_test_logic_path": str(cloudn_full_path),
            "error": None,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        error_result = {
            "validation_result": "Error",
            "analysis": "",
            "expected_test_logic_path": str(regression_full_path),
            "actual_test_logic_path": str(cloudn_full_path),
            "error": (
                f"Error validating Python test logic: {str(e)}. "
                + "Please check if the test path is correct. "
                + f"Regression path {regression_full_path} "
                + f"or cloudn path {cloudn_full_path} is not correct."
            ),
        }
        return json.dumps(error_result, indent=2)


@mcp.tool(name="prepare_terraform_syntax_validation")
async def prepare_terraform_syntax_validation(
    ctx: Context,
    test_suite_name: str,
) -> str:
    """
    Use this tool to prepare the testing environment for the input test suite Terraform topology,
    and get validation command to execute and where to run the command.
    Please run the command in the given directory. And verify the result.

    Args:
        test_suite_name: The name of the test suite to execute

    Returns:
        JSON string containing test execution information with structure and field meanings:

        Structure:
        {
            "execution_path": Directory where you should run the command,
            "execution_command": The exact command to run for Terraform syntax validation,
            "notes": Additional guidance or caveats for running the test,
            "error": Error message if preparation failed.
        }
    """

    test_execution_dir = ctx.request_context.lifespan_context.test_execution_dir

    # Set working directory
    relative_test_dir = f"tests/{test_suite_name}"
    test_dir = test_execution_dir / relative_test_dir

    # Prepare commands
    cp_command = f"cp {TFVARS_PATH} {test_dir}/{TFVARS_FILE_NAME}"
    run_command = (
        f"terraform init && terraform apply -auto-approve -var-file={TFVARS_FILE_NAME}"
    )

    # Execute credential file copy command
    try:
        print(f"📋 Executing: {cp_command}")
        subprocess.run(
            cp_command, shell=True, capture_output=True, text=True, check=True
        )
        await ctx.info(f"Credential file copied for test: {test_suite_name}")
    except subprocess.CalledProcessError as e:
        error_result: dict[str, str] = {
            "error": f"Error copying credential file: {e.stderr or str(e)}",
        }
        return json.dumps(error_result, indent=2)

    # Prepare response with execution information in JSON format
    result: dict[str, str] = {
        "execution_path": str(test_dir),
        "execution_command": run_command,
        "notes": (
            f"{TFVARS_FILE_NAME} is a provider credential file "
            + "provided by the test framework and should not be modified."
            + "If needed, update Terraform variable names in the test "
            + "script to match the provider credential file."
        ),
    }

    await ctx.info(
        f"Terraform topology validation environment prepared for: {test_suite_name}"
    )
    return json.dumps(result, indent=2)


@mcp.tool(name="entire_test_validation_preparation")
async def prepare_entire_test_syntax_validation(
    ctx: Context,
    test_suite_name: str,
) -> str:
    """
    Use this tool to prepare the testing environment for the input test suite,
    and get execution command to execute and where to run the command.
    Please run the command in the given directory. And verify the result.

    Args:
        test_suite_name: The name of the test suite to execute
    Returns:
        JSON string containing test execution information with structure and field meanings:

        Structure:
        {
            "execution_path": Directory where you should run the command,
            "execution_command": The exact command to run for the full test run,
            "logs_directory": Directory where complete test execution logs will be written,
            "report_path": Path to the generated HTML report for the full test run,
            "notes": Additional guidance or caveats for running the test,
            "error": Error message if preparation failed.
        }
    """
    test_execution_dir = ctx.request_context.lifespan_context.test_execution_dir
    log_dir = ctx.request_context.lifespan_context.log_dir

    # Validate inputs
    if not test_suite_name or not test_suite_name.strip():
        input_error: dict[str, str] = {
            "error": "Test name cannot be empty",
        }
        return json.dumps(input_error, indent=2)

    # Set working directory
    relative_test_dir = f"tests/{test_suite_name}"
    test_dir = test_execution_dir / relative_test_dir

    # Prepare commands
    cp_command = f"cp {TFVARS_PATH} {test_dir}/{TFVARS_FILE_NAME}"
    run_command = f"export AVX_NODESTROY=1 && uv run pytest {relative_test_dir} -sv --log-cli-level=INFO"

    # Execute credential file copy command
    try:
        print(f"📋 Executing: {cp_command}")
        subprocess.run(
            cp_command, shell=True, capture_output=True, text=True, check=True
        )
        await ctx.info(f"Credential file copied for test: {test_suite_name}")
    except subprocess.CalledProcessError as e:
        copy_error: dict[str, str] = {
            "error": f"Error copying credential file: {e.stderr or str(e)}",
        }
        return json.dumps(copy_error, indent=2)

    # Prepare response with execution information in JSON format
    result: dict[str, str] = {
        "execution_path": str(test_dir),
        "execution_command": run_command,
        "logs_directory": str(log_dir),
        "report_path": str(test_execution_dir / "report.html"),
        "notes": (
            f"{TFVARS_FILE_NAME} is a provider credential file "
            + "provided by the test framework and should not be modified."
            + "If needed, update Terraform variable names in the test "
            + "script to match the provider credential file."
        ),
    }

    await ctx.info(f"Test environment prepared for: {test_suite_name}")
    return json.dumps(result, indent=2)


async def main():
    """
    Main entry point for the MCP server.
    """

    if not Path(CLOUDN_PATH).exists():
        logger.error("CLOUDN_PATH environment variable is required but not set")
        logger.error("Please set CLOUDN_PATH in your .env file")
        raise RuntimeError("CLOUDN_PATH environment variable is required but not set")

    if not TFVARS_PATH or not Path(TFVARS_PATH).exists():
        logger.error("TFVARS_PATH environment variable is required but not set")
        logger.error("Please set TFVARS_PATH in your .env file")
        raise RuntimeError("TFVARS_PATH environment variable is required but not set")

    if not LLM_API_KEY:
        logger.error("LLM_API_KEY environment variable is required but not set")
        logger.error("Please set LLM_API_KEY in your .env file")
        raise RuntimeError("LLM_API_KEY environment variable is required but not set")

    mcp_tools = await mcp.list_tools()
    print("📦 Registered tools:", [t.name for t in mcp_tools])

    print("🚀 E2E Migration MCP server starting using SSE transport")
    print(f"🤖 LLM Model: {LLM_CHAT_MODEL_NAME}")
    print(f"🔗 LLM Base URL: {LLM_BASE_URL}")
    await mcp.run_sse_async()

    print(f"CLOUDN_PATH: {CLOUDN_PATH}")
    print(f"TFVARS_PATH: {TFVARS_PATH}")
    print(f"LLM_MODEL: {LLM_CHAT_MODEL_NAME}")


if __name__ == "__main__":
    asyncio.run(main())
