# AGENTS.md

## Code Generation

Make sure you adhere to the guide on this file to generate the code.

### Rules of engagement

1. You're allowed to create functions, structs and impl blocks to generate the code.
2. For function signatures, include the type from `typing` package. Optional arguments should be put after mandatory arguments. 
3. With the exception of import statements, explain the flow of the program using comments. Make sure that you write not only what the program is doing, but why. It will help me to judge your work.
4. You have to add documentation for new functions and classes that you generate. For existing functions, update the documentation for functions and classes that you edit. The documentation should follow NumPy style. The documentation should contain what is the function for (basically the description), a brief summary of the steps, and input and output parameters. If your function has the ability to throw error, please state it in the documentation as well.
5. You have to add unit tests as well. More on this in the next subchapter.

### Testing Standards

1. Make sure to put unit tests in `tests` folder.
2. **Mandatory Coverage**: Every new feature or bug fix must include corresponding unit tests.
3. **Framework**: Use `pytest`. Prefer functional tests with fixtures over heavily mocked class-based tests.
4. **Async Code**: If testing async functions, use `@pytest.mark.anyio` or `@pytest.mark.asyncio` as appropriate for the project stack.
5. **Mocking**: Use `pytest-mock` (mocker fixture) instead of standard `unittest.mock` where possible for cleaner teardowns.

### Debugging

In this repository, we don't have Python installed in local machine. Instead, Python executable is managed by Pixi, which is why in [pixi.toml](pixi.toml) under `[dependencies]` section you will have `python` listed as dependency.

The consequence is that every Python command you want to run should be run under `pixi run` command. I will list some of the examples here:
- Check Python version: Instead of `python3 --version`, you should run `pixi run python3 --version`.

## Code Validation

Make sure you validate the code you generated.

### Steps to Validate

Run these commands in sequence:
1. `pixi run lint-ci`: ensure no linter errors.
2. `pixi run lint-fix`: if there is any linter error, fix it with this command.
3. `pixi run lint-ci`: recheck again, maybe there are linter errors that need manual fix.
4. `pixi run type-check`: ensure no type errors.
5. `pixi run test`: build the code, then run all unit tests. This ensures that the code you generate pass all the unit tests.

## Update Documentation

After the validation is finished, update the project tree structure and file descriptions in README.md if needed. This is to ensure we always have updated documentation.

## Tooling

You're allowed to use MCP servers provided in `mcp.json`.
