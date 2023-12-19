import llm
import os
import sys
import click
import json
from typing import Optional, List, Tuple
from huggingface_hub import snapshot_download

try:
    from pydantic import Field, field_validator  # type: ignore
except ImportError:
    from pydantic.class_validators import (
        validator as field_validator,
    )  # type: ignore [no-redef]
    from pydantic.fields import Field

def _ensure_hf_home():
    hf_home = llm.user_dir() / "huggingface_local" / "hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    return str(hf_home.resolve())

def _ensure_models_file():
    plugin_dir = llm.user_dir() / "huggingface_local"
    filepath = plugin_dir / "models.json"
    if not filepath.exists():
        filepath.write_text("{}")
    return filepath

def build_prompt_blocks_and_system(
    self, prompt: llm.Prompt, conversation: Optional[llm.Conversation]
) -> Tuple[List[str], str]:
    blocks = []

    # Simplified handling of system prompts: use the one from prompt.system, or the
    # one from the first message in the conversation, or the default for the model.
    # Ignore the case where the system prompt changed mid-conversation.
    system_prompt = None
    if prompt.system:
        system_prompt = prompt.system

    if conversation is not None:
        for response in conversation.responses:
            if response.prompt.system:
                system_prompt = response.prompt.system
                break

    if system_prompt is None:
        system_prompt = self.system_prompt()

    template = self.prompt_template()
    # Special case to add <|im_end|> if it looks necessary
    template_end = ""
    if "<|im_start|>" in template and template.count(
        "<|im_start|>"
    ) - 1 == template.count("<|im_end|>"):
        template_end = "<|im_end|>"

    if conversation is not None:
        for prev_response in conversation.responses:
            blocks.append(template.replace("%1", prev_response.prompt.prompt))
            blocks.append(prev_response.text() + template_end)

    # Add the user's prompt
    blocks.append(template.replace("%1", prompt.prompt))

    return blocks, system_prompt

@llm.hookimpl
def register_models(register):
    plugin_dir = llm.user_dir() / "huggingface_local"
    models_file = plugin_dir / "models.json"
    if not models_file.exists():
        return
    models = json.loads(models_file.read_text())
    for model_id, details in models.items():
        register(
            HuggingfaceModel(
                model_id
            ),
            aliases=details["aliases"],
        )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def llm_huggingface_local():
        "Commands for registering Huggingface local models with LLM"

    @llm_huggingface_local.command()
    def models_file():
        "Display the path to the models.json file"
        plugin_dir = llm.user_dir() / "huggingface_local"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        models_file = plugin_dir / "models.json"
        click.echo(models_file)

    @llm_huggingface_local.command()
    def models_dir():
        "Display the path to the directory holding downloaded models"
        plugin_dir = llm.user_dir() / "huggingface_local"
        click.echo(plugin_dir)

    @llm_huggingface_local.command()
    @click.argument(
        "name"
    )
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def add_model(name, aliases):
        "Register a Huggingface model"
        import transformers
        transformers.logging.set_verbosity_error()
        hf_home = _ensure_hf_home()
        #Validate model exists and download
        #snapshot_download(repo_id=name, cache_dir=hf_home)
        transformers.pipeline("text-generation", model=name, model_kwargs={'cache_dir':hf_home})
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        info = {
            "name": name,
            "aliases": aliases
        }
        models[name] = info
        models_file.write_text(json.dumps(models, indent=2))

    @llm_huggingface_local.command()
    def models():
        "List registered Huggingface models"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        click.echo(json.dumps(models, indent=2))


class HuggingfaceModel(llm.Model):
    class Options(llm.Options):
        verbose: bool = Field(
            description="Whether to print verbose output from the model", default=False
        )
        max_tokens: int = Field(
            description="Max tokens to return, defaults to 4000", default=4000
        )

    def __init__(self, model_id):
            self.model_id = model_id
            self.pipe = None

    def execute(self, prompt, stream, response, conversation):
        with SuppressOutput():
            hf_home = _ensure_hf_home()
            import transformers
            transformers.logging.set_verbosity_error()
            transformers.logging.disable_progress_bar()
            if self.pipe == None:
                self.pipe = transformers.pipeline("text-generation", model=self.model_id, model_kwargs={'cache_dir':hf_home})
            yield self.pipe(prompt.prompt, max_new_tokens=prompt.options.max_tokens)[0]['generated_text']


class SuppressOutput:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            return
        # Save a copy of the current file descriptors for stdout and stderr
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)

        # Open a file to /dev/null
        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)

        # Replace stdout and stderr with /dev/null
        os.dup2(self.devnull_fd, 1)
        os.dup2(self.devnull_fd, 2)

        # Writes to sys.stdout and sys.stderr should still work
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = os.fdopen(self.stdout_fd, "w")
        sys.stderr = os.fdopen(self.stderr_fd, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            return
        # Restore stdout and stderr to their original state
        os.dup2(self.stdout_fd, 1)
        os.dup2(self.stderr_fd, 2)

        # Close the saved copies of the original stdout and stderr file descriptors
        os.close(self.stdout_fd)
        os.close(self.stderr_fd)

        # Close the file descriptor for /dev/null
        os.close(self.devnull_fd)

        # Restore sys.stdout and sys.stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr