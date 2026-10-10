"""
Reusable prompt template system with pre-built templates and CLI interface.

Provides a flexible PromptTemplate class for creating parameterized prompts,
a TemplateRegistry for managing templates, and pre-built templates for common
LLM tasks like summarization, translation, and code review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PromptTemplate:
    """A parameterized prompt template that supports variable substitution.

    Attributes:
        name: Unique name identifier for the template.
        template: The template string with {variable} placeholders.
        description: Human-readable description of what the template does.
        required_vars: Set of variable names that must be provided.
        optional_vars: Mapping of variable names to default values.
    """

    name: str
    template: str
    description: str = ""
    required_vars: set[str] = field(default_factory=set)
    optional_vars: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Extract variable names from template if not explicitly provided."""
        found_vars = set(re.findall(r"\{(\w+)\}", self.template))
        if not self.required_vars:
            self.required_vars = found_vars - set(self.optional_vars.keys())

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variable values.

        Args:
            **kwargs: Variable values to substitute into the template.

        Returns:
            The rendered template string.

        Raises:
            ValueError: If a required variable is missing.
        """
        missing = self.required_vars - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {', '.join(sorted(missing))}")

        merged = {**self.optional_vars, **kwargs}
        result = self.template
        for key, value in merged.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def extract_variables(self) -> list[str]:
        """Return all variable names found in the template."""
        return sorted(set(re.findall(r"\{(\w+)\}", self.template)))


class TemplateRegistry:
    """A registry for storing, retrieving, and managing prompt templates.

    Provides methods to register templates, retrieve them by name,
    list all available templates, and export/import template collections.
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register a template in the registry.

        Args:
            template: The PromptTemplate to register.

        Raises:
            ValueError: If a template with the same name already exists.
        """
        if template.name in self._templates:
            raise ValueError(f"Template '{template.name}' already exists")
        self._templates[template.name] = template

    def unregister(self, name: str) -> None:
        """Remove a template from the registry.

        Args:
            name: Name of the template to remove.

        Raises:
            KeyError: If the template does not exist.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")
        del self._templates[name]

    def get(self, name: str) -> PromptTemplate:
        """Retrieve a template by name.

        Args:
            name: Name of the template.

        Returns:
            The PromptTemplate with the given name.

        Raises:
            KeyError: If the template does not exist.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found. Available: {', '.join(self.list_names())}")
        return self._templates[name]

    def list_names(self) -> list[str]:
        """Return sorted list of all registered template names."""
        return sorted(self._templates.keys())

    def list_templates(self) -> list[PromptTemplate]:
        """Return list of all registered templates."""
        return list(self._templates.values())

    def export_json(self) -> str:
        """Export all templates as a JSON string."""
        data = []
        for t in self._templates.values():
            data.append({
                "name": t.name,
                "template": t.template,
                "description": t.description,
                "required_vars": sorted(t.required_vars),
                "optional_vars": t.optional_vars,
            })
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> int:
        """Import templates from a JSON string.

        Args:
            json_str: JSON string containing template definitions.

        Returns:
            Number of templates imported.
        """
        data = json.loads(json_str)
        count = 0
        for item in data:
            template = PromptTemplate(
                name=item["name"],
                template=item["template"],
                description=item.get("description", ""),
                required_vars=set(item.get("required_vars", [])),
                optional_vars=item.get("optional_vars", {}),
            )
            if template.name not in self._templates:
                self.register(template)
                count += 1
        return count


def create_default_registry() -> TemplateRegistry:
    """Create a registry populated with pre-built templates.

    Returns:
        A TemplateRegistry containing standard prompt templates.
    """
    registry = TemplateRegistry()

    registry.register(PromptTemplate(
        name="summarization",
        template=(
            "Please provide a concise summary of the following text. "
            "The summary should capture the key points and main ideas.\n\n"
            "Text:\n{text}\n\n"
            "{optional_instructions}"
        ),
        description="Summarize a given text into key points.",
        required_vars={"text"},
        optional_vars={
            "optional_instructions": "Keep the summary under 3 sentences."
        },
    ))

    registry.register(PromptTemplate(
        name="translation",
        template=(
            "Translate the following text from {source_language} to {target_language}. "
            "Maintain the original tone and meaning.\n\n"
            "Text to translate:\n{text}"
        ),
        description="Translate text between languages.",
        required_vars={"text", "target_language"},
        optional_vars={"source_language": "English"},
    ))

    registry.register(PromptTemplate(
        name="code_review",
        template=(
            "You are an expert code reviewer. Review the following {language} code "
            "and provide constructive feedback.\n\n"
            "Code:\n```{language}\n{code}\n```\n\n"
            "Please analyze:\n"
            "1. Code correctness and potential bugs\n"
            "2. Performance optimizations\n"
            "3. Code style and best practices\n"
            "4. Security concerns\n\n"
            "Focus on: {focus_areas}"
        ),
        description="Review code and provide constructive feedback.",
        required_vars={"code", "language"},
        optional_vars={"focus_areas": "all areas"},
    ))

    registry.register(PromptTemplate(
        name="email_writing",
        template=(
            "Write a {tone} email with the following details:\n\n"
            "To: {recipient}\n"
            "Subject: {subject}\n"
            "Purpose: {purpose}\n\n"
            "Additional context: {context}\n\n"
            "Ensure the email is professional, clear, and well-structured."
        ),
        description="Write a professional email with specified tone and purpose.",
        required_vars={"recipient", "subject", "purpose"},
        optional_vars={
            "tone": "professional",
            "context": "No additional context provided.",
        },
    ))

    registry.register(PromptTemplate(
        name="data_extraction",
        template=(
            "Extract structured data from the following text and return it as JSON.\n\n"
            "Text:\n{text}\n\n"
            "Expected fields to extract:\n{fields}\n\n"
            "Return a valid JSON object with the extracted fields. "
            "If a field is not found in the text, use null as the value."
        ),
        description="Extract structured data from unstructured text.",
        required_vars={"text", "fields"},
    ))

    registry.register(PromptTemplate(
        name="explain_concept",
        template=(
            "Explain the concept of {topic} to a {audience}.\n\n"
            "Requirements:\n"
            "- Use simple, clear language\n"
            "- Provide practical examples\n"
            "- Keep it under {max_words} words\n\n"
            "{additional_instructions}"
        ),
        description="Explain a concept tailored to a specific audience.",
        required_vars={"topic", "audience"},
        optional_vars={
            "max_words": "300",
            "additional_instructions": "",
        },
    ))

    registry.register(PromptTemplate(
        name="bullet_points",
        template=(
            "Convert the following text into clear, organized bullet points.\n\n"
            "Text:\n{text}\n\n"
            "Format each point as a concise statement. Group related items under "
            "clear headings if appropriate."
        ),
        description="Convert text into organized bullet points.",
        required_vars={"text"},
    ))

    registry.register(PromptTemplate(
        name="pros_cons",
        template=(
            "Analyze the pros and cons of: {topic}\n\n"
            "Provide a balanced analysis with:\n"
            "- At least 3 pros with explanations\n"
            "- At least 3 cons with explanations\n"
            "- A brief overall assessment\n\n"
            "Context: {context}"
        ),
        description="Provide a balanced pros and cons analysis.",
        required_vars={"topic"},
        optional_vars={"context": ""},
    ))

    return registry


def interactive_cli() -> None:
    """Run an interactive CLI for using prompt templates."""
    registry = create_default_registry()

    parser = argparse.ArgumentParser(
        description="Prompt Template System - Create and use LLM prompt templates"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("list", help="List all available templates")

    use_parser = subparsers.add_parser("use", help="Use a template")
    use_parser.add_argument("template_name", help="Name of the template to use")
    use_parser.add_argument("--vars", nargs="*", help="Variable assignments (key=value)")
    use_parser.add_argument("--json-input", action="store_true", help="Read variables from stdin as JSON")

    show_parser = subparsers.add_parser("show", help="Show template details")
    show_parser.add_argument("template_name", help="Name of the template")

    export_parser = subparsers.add_parser("export", help="Export templates to JSON")
    export_parser.add_argument("--output", "-o", help="Output file path (default: stdout)")

    import_parser = subparsers.add_parser("import", help="Import templates from JSON")
    import_parser.add_argument("input_file", help="JSON file to import")

    vars_parser = subparsers.add_parser("vars", help="Show variables for a template")
    vars_parser.add_argument("template_name", help="Name of the template")

    args = parser.parse_args()

    if args.command == "list":
        print("\nAvailable templates:")
        print("-" * 50)
        for tmpl in registry.list_templates():
            print(f"  {tmpl.name:<20} {tmpl.description}")
        print()

    elif args.command == "show":
        try:
            tmpl = registry.get(args.template_name)
            print(f"\nTemplate: {tmpl.name}")
            print(f"Description: {tmpl.description}")
            print(f"Required vars: {sorted(tmpl.required_vars)}")
            print(f"Optional vars: {tmpl.optional_vars}")
            print(f"\nTemplate text:\n{tmpl.template}\n")
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "use":
        try:
            tmpl = registry.get(args.template_name)
            variables: dict[str, str] = {}

            if args.json_input:
                variables = json.load(sys.stdin)
            elif args.vars:
                for item in args.vars:
                    if "=" not in item:
                        print(f"Warning: skipping invalid format '{item}'", file=sys.stderr)
                        continue
                    key, value = item.split("=", 1)
                    variables[key.strip()] = value.strip()

            result = tmpl.render(**variables)
            print(f"\n{'=' * 60}")
            print("RENDERED PROMPT:")
            print("=" * 60)
            print(result)
            print("=" * 60)
        except (ValueError, KeyError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "export":
        json_str = registry.export_json()
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"Exported {len(registry.list_names())} templates to {args.output}")
        else:
            print(json_str)

    elif args.command == "import":
        try:
            with open(args.input_file, encoding="utf-8") as f:
                count = registry.import_json(f.read())
            print(f"Imported {count} new templates")
        except FileNotFoundError:
            print(f"Error: File '{args.input_file}' not found", file=sys.stderr)
            sys.exit(1)

    elif args.command == "vars":
        try:
            tmpl = registry.get(args.template_name)
            print(f"\nVariables for '{tmpl.name}':")
            for var in sorted(tmpl.extract_variables()):
                status = "required" if var in tmpl.required_vars else f"optional (default: {tmpl.optional_vars.get(var)})"
                print(f"  {{{var}}} - {status}")
            print()
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    interactive_cli()
