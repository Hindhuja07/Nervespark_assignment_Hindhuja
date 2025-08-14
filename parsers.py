import re

EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp",
    ".c": "c", ".h": "c", ".md": "markdown", ".rst": "rst"
}

def lang_from_filename(filename: str):
    for ext, lang in EXT_LANG.items():
        if filename.endswith(ext):
            return lang
    return "text"

# simple regex function/class splitters per language
FUNC_RE = re.compile(r"^\s*(def |class |function |func |public |private |export )", re.MULTILINE)
HEAD_RE = re.compile(r"(^#{1,6}\s+.*$)|(^\s*\\/\\/{1,}.*$)", re.MULTILINE)
