"""Evidence-based, non-promoting classification and reporting for pytest inventory."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from functools import lru_cache
import fnmatch
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any, Mapping, Sequence

from turbo_regression.capabilities import SUPPORTED
from turbo_regression.classification import (
    ClassificationError,
    FEATURES,
    classify_test,
    load_manifest,
    stable_test_id,
)

LEVELS = ("VERIFIED", "INFERRED", "UNKNOWN")
BUILTIN_FIXTURES = frozenset({
    "cache", "capfd", "capfdbinary", "caplog", "capsys", "capsysbinary",
    "doctest_namespace", "monkeypatch", "pytestconfig", "record_property",
    "record_testsuite_property", "record_xml_attribute", "recwarn", "request",
    "tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory",
})
PLATFORM_IMPORTS = {
    "Windows": {"winreg", "msvcrt", "win32api", "win32con", "win32file", "win32process"},
    "Linux": {"fcntl", "pwd", "grp", "termios", "pty", "systemd"},
}
STATIC_RESOURCE_RULES = {
    "fixed_tcp_port": ("socket.bind", "fixed port"),
    "filesystem_lock": ("fcntl.flock", "file lock"),
    "global_environment": ("os.environ", "environment mutation"),
    "shared_database": ("sqlite3.connect", "database"),
    "browser_profile": ("launch_persistent_context", "browser profile"),
    "docker_resource": ("docker", "Docker resource"),
    "external_service": ("requests.", "external service"),
}


def load_validation_manifest(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        manifest: Any = {"schema_version": 1, "validations": {}}
    elif isinstance(value, str):
        with open(value, encoding="utf-8") as stream:
            manifest = json.load(stream)
    else:
        manifest = dict(value)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or set(manifest) - {"schema_version", "validations"}
        or not isinstance(manifest.get("validations", {}), dict)
    ):
        raise ClassificationError("validation manifest must use schema_version 1 and a validations object")
    for selector, entry in manifest["validations"].items():
        if not isinstance(selector, str) or not isinstance(entry, dict):
            raise ClassificationError("validation entries must map selectors to objects")
        if set(entry) - {"requirements_sha256", "targets"}:
            raise ClassificationError(f"unknown validation fields for {selector}")
        targets = entry.get("targets", {})
        if not isinstance(targets, dict):
            raise ClassificationError(f"validation targets must be an object for {selector}")
        for target, evidence in targets.items():
            if (
                not isinstance(target, str) or not isinstance(evidence, dict)
                or set(evidence) - {"result", "source_commit_sha", "run_id", "evidence"}
            ):
                raise ClassificationError(f"invalid target validation for {selector}")
    return {"schema_version": 1, "validations": manifest.get("validations", {})}


@lru_cache(maxsize=1024)
def _parse_file(filename: str, modified_ns: int, size: int) -> tuple[ast.AST | None, str]:
    try:
        text = Path(filename).read_text(encoding="utf-8")
        return ast.parse(text, filename=filename), text
    except (OSError, UnicodeError, SyntaxError):
        return None, ""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _fixture_closure(record: Mapping[str, Any]) -> tuple[set[str], list[dict], list[str]]:
    graph = record.get("fixture_graph", {})
    if not isinstance(graph, Mapping):
        return set(record.get("fixtures", [])), [], ["fixture graph is missing or malformed"]
    unresolved: list[str] = []
    closure: set[str] = set()
    fixture_names: set[str] = set()
    visiting: list[str] = []
    details_by_id: dict[str, dict] = {}

    def visit(reference: str) -> None:
        if reference in visiting:
            cycle = visiting[visiting.index(reference):] + [reference]
            names = [
                str((graph.get(item) or {}).get("fixture_name", item))
                for item in cycle
            ]
            unresolved.append("fixture dependency cycle: " + " -> ".join(names))
            return
        if reference in closure:
            return
        definition = graph.get(reference)
        if not isinstance(definition, Mapping):
            if reference in BUILTIN_FIXTURES:
                fixture_names.add(reference)
                closure.add(reference)
            else:
                unresolved.append(f"unresolved fixture: {reference}")
                closure.add(reference)
            return
        name = str(definition.get("fixture_name", reference))
        fixture_names.add(name)
        details_by_id[reference] = {"fixture": name, "metadata": dict(definition)}
        if definition.get("resolved") is not True:
            unresolved.append(f"unresolved fixture: {name}")
        if definition.get("dynamic_lookup") is True:
            unresolved.append(f"dynamic fixture lookup: {name}")

        bindings = definition.get("dependency_bindings")
        if isinstance(bindings, Mapping):
            children = list(bindings.values())
        else:
            dependencies = definition.get("dependencies", [])
            dynamic_dependencies = definition.get("dynamic_dependencies", [])
            if not isinstance(dependencies, list) or not isinstance(dynamic_dependencies, list):
                unresolved.append(f"invalid dependency metadata: {name}")
                dependencies, dynamic_dependencies = [], []
            children = [*dependencies, *dynamic_dependencies]

        visiting.append(reference)
        for child in children:
            if isinstance(child, str):
                visit(child)
            else:
                unresolved.append(f"invalid fixture dependency from {name}")
        visiting.pop()
        closure.add(reference)

    roots = record.get("fixture_roots")
    if isinstance(roots, Mapping):
        root_references = roots.values()
        for reference in root_references:
            if isinstance(reference, str):
                visit(reference)
            else:
                unresolved.append("invalid fixture root reference")
    else:
        for fixture in record.get("fixtures", []):
            if isinstance(fixture, str):
                visit(fixture)
            else:
                unresolved.append(str(fixture))
    return fixture_names, [
        details_by_id[key] for key in sorted(details_by_id)
    ], sorted(set(unresolved))


def _source_paths(record: Mapping[str, Any], graph_details: Sequence[Mapping[str, Any]]) -> list[str]:
    paths = {record.get("source_file")}
    for dependency in record.get("source_dependencies", []):
        if isinstance(dependency, Mapping):
            paths.add(dependency.get("source_file"))
    for entry in graph_details:
        metadata = entry.get("metadata")
        if isinstance(metadata, Mapping):
            paths.add(metadata.get("source_file"))
    return sorted(path for path in paths if isinstance(path, str) and Path(path).is_file())


@lru_cache(maxsize=2048)
def _cached_static_evidence(
    signatures: tuple[tuple[str, int, int], ...],
    platform_clues: tuple[str, ...],
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    evidence, resources = _static_evidence_uncached(
        [filename for filename, _, _ in signatures], platform_clues,
    )
    return tuple(evidence), tuple(resources)


def _static_evidence(paths: Sequence[str], platform_clues: Sequence[str]) -> tuple[list[dict], list[dict]]:
    signatures = []
    for filename in paths:
        try:
            stat = Path(filename).stat()
            signatures.append((filename, stat.st_mtime_ns, stat.st_size))
        except OSError:
            signatures.append((filename, -1, -1))
    evidence, resources = _cached_static_evidence(tuple(signatures), tuple(platform_clues))
    return list(evidence), list(resources)


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _private_sqlite(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {":memory:", ""} or lowered.startswith("file::memory:")


def _private_path(node: ast.AST) -> bool:
    """A path that pytest builds from a per-test temporary directory."""
    if isinstance(node, ast.Name) and node.id in {"tmp_path", "tmpdir"}:
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _private_path(node.left)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"joinpath", "resolve"}:
            return _private_path(node.func.value)
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"Path", "PurePath", "PurePosixPath", "PureWindowsPath"}
        and node.args
    ):
        return _private_path(node.args[0])
    return False


def _postgres_name(value: str) -> str | None:
    from urllib.parse import urlparse

    if "://" not in value:
        return None
    parsed = urlparse(value)
    name = parsed.path.lstrip("/")
    if not parsed.hostname or not name or "/" in name:
        return None
    return name


def _http_origin(value: str) -> str | None:
    from urllib.parse import urlparse

    if not value.lower().startswith(("http://", "https://")):
        return None
    parsed = urlparse(value)
    if not parsed.hostname:
        return None
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _environ_target(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "environ"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
    )


def _resource(filename: str, resource: str, scope: str, detail: str, *, exclusive: bool = False, wildcard: bool = False) -> dict:
    item = {
        "resource": resource, "file": filename, "scope": scope,
        "detail": detail, "confidence": "INFERRED",
    }
    if exclusive:
        item["batch_exclusive"] = True
    if wildcard:
        item["wildcard"] = True
    return item


def _merge_kind(left: str | None, right: str) -> str:
    if left is None or left == right:
        return right
    return "unresolved"


def _database_constant(value: str) -> str | None:
    """A concrete database path. Private memory targets are not a lock."""
    if _private_sqlite(value):
        return "private"
    return "literal:" + value


def _parameter_kinds(func: ast.AST) -> dict[str, str]:
    """tmp_path is per-test. A parameter with no default is caller-chosen.

    A string default may be a shared file, so a direct connect of that parameter
    stays unresolved. Assigning it to ``self`` is handled separately.
    """
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    kinds = {}
    for arg, default in zip(positional, defaults):
        kinds[arg.arg] = _parameter_kind(arg.arg, default)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        kinds[arg.arg] = _parameter_kind(arg.arg, default)
    return kinds


def _parameter_kind(name: str, default: ast.AST | None) -> str:
    if name in {"tmp_path", "tmpdir", "tmp_path_factory"}:
        return "private"
    if isinstance(default, ast.Constant) and isinstance(default.value, str):
        if _database_constant(default.value) != "private":
            return "unresolved"
    return "caller"


def _expression_kind(node: ast.AST, names: Mapping[str, str], attributes: Mapping[str, str]) -> str:
    if _private_path(node):
        return "private"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _database_constant(node.value) or "unresolved"
    if isinstance(node, ast.Name):
        return names.get(node.id, "unresolved")
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"self", "cls"}
    ):
        return attributes.get(node.attr, "unresolved")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
        "str", "Path", "PurePath", "PurePosixPath", "PureWindowsPath",
    } and node.args:
        return _expression_kind(node.args[0], names, attributes)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id not in {"self", "cls"}
    ):
        owner = names.get(node.value.id, "unresolved")
        return owner if owner in {"private", "parameterized", "caller"} else "unresolved"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
        "joinpath", "resolve", "as_uri", "expanduser", "absolute", "as_posix",
    }:
        kind = _expression_kind(node.func.value, names, attributes)
        return kind if kind in {"private", "parameterized", "caller"} else "unresolved"
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        kind = _expression_kind(node.left, names, attributes)
        if kind in {"private", "parameterized", "caller"} and isinstance(node.right, ast.Constant):
            return kind
    if isinstance(node, ast.IfExp):
        return _merge_kind(
            _expression_kind(node.body, names, attributes),
            _expression_kind(node.orelse, names, attributes),
        )
    if isinstance(node, ast.JoinedStr):
        kinds = [
            _expression_kind(value.value, names, attributes)
            for value in node.values
            if isinstance(value, ast.FormattedValue)
        ]
        if kinds and all(kind in {"private", "parameterized", "caller"} for kind in kinds):
            return kinds[0] if all(kind == kinds[0] for kind in kinds) else "unresolved"
    return "unresolved"


def _owned_statements(node: ast.AST) -> list[ast.stmt]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
        return list(node.body)
    return []


def _assignments_in(statements: Sequence[ast.stmt], names: list[tuple[str, ast.AST]], attributes: list[tuple[str, ast.AST]]) -> None:
    for statement in statements:
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets = [statement.target]
            value = statement.value
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if value is not None:
            for target in targets:
                if isinstance(target, ast.Name):
                    names.append((target.id, value))
                elif (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {"self", "cls"}
                ):
                    attributes.append((target.attr, value))
        for child in (
            "body", "orelse", "finalbody",
        ):
            nested = getattr(statement, child, None)
            if isinstance(nested, list):
                _assignments_in([item for item in nested if isinstance(item, ast.stmt)], names, attributes)
        if isinstance(statement, ast.Try):
            for handler in statement.handlers:
                _assignments_in(handler.body, names, attributes)


def _solve_kinds(pairs: Sequence[tuple[str, ast.AST]], initial: Mapping[str, str], attributes: Mapping[str, str]) -> dict[str, str]:
    kinds = dict(initial)
    for _ in range(8):
        changed = False
        for name, value in pairs:
            kind = _expression_kind(value, kinds, attributes)
            merged = _merge_kind(kinds.get(name), kind)
            if kinds.get(name) != merged:
                kinds[name] = merged
                changed = True
        if not changed:
            break
    return kinds


def _connect_kind(kind: str, function_name: str) -> str:
    """A test parameter may be a shared fixture. Any other parameter is the caller's path."""
    if kind == "caller":
        return "unresolved" if function_name.startswith("test_") else "parameterized"
    return kind


def _attribute_kind(kind: str) -> str:
    """Storing a caller-chosen parameter on ``self`` does not pick a shared file."""
    if kind == "caller":
        return "parameterized"
    return kind


def _is_property(function: ast.AST) -> bool:
    return any(
        (isinstance(item, ast.Name) and item.id == "property")
        or (isinstance(item, ast.Attribute) and item.attr == "property")
        for item in getattr(function, "decorator_list", ())
    )


def _return_values(function: ast.AST) -> list[ast.AST]:
    found: list[ast.AST] = []

    def visit(statements: Sequence[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, ast.Return) and statement.value is not None:
                found.append(statement.value)
            for child in ("body", "orelse", "finalbody"):
                nested = getattr(statement, child, None)
                if isinstance(nested, list):
                    visit([item for item in nested if isinstance(item, ast.stmt)])
            if isinstance(statement, ast.Try):
                for handler in statement.handlers:
                    visit(handler.body)

    visit(getattr(function, "body", []))
    return found


def _database_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    return _dotted_name(node.func) in {"sqlite3.connect", "psycopg.connect", "psycopg2.connect"}


def _call_argument(node: ast.Call) -> ast.AST | None:
    if node.args:
        return node.args[0]
    for keyword in node.keywords:
        if keyword.arg in {None, "database", "dsn", "dbname"}:
            return keyword.value
    return None


def _direct_database_literals(tree: ast.AST) -> set[str]:
    """A string path passed into a call. A path joined onto tmp_path is not one."""
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        arguments = list(node.args)
        arguments.extend(keyword.value for keyword in node.keywords)
        for argument in arguments:
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                continue
            kind = _database_constant(argument.value)
            filename_text = argument.value.replace("\\", "/").split("?", 1)[0].rsplit("/", 1)[-1].lower()
            if (
                kind
                and kind.startswith("literal:")
                and "*" not in filename_text
                and "." in filename_text
                and filename_text.rsplit(".", 1)[-1] in {"sqlite", "sqlite3", "db"}
            ):
                found.add("shared_database:" + argument.value)
    return found


def _database_evidence(tree: ast.AST, text: str, calls: set[str], filename: str) -> list[dict]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    class_attributes: dict[int, dict[str, str]] = {}
    function_names: dict[int, dict[str, str]] = {}
    for function in functions:
        function_names[id(function)] = _parameter_kinds(function)
    for _ in range(8):
        changed = False
        for function in functions:
            owner = parents.get(function)
            while owner is not None and not isinstance(owner, ast.ClassDef):
                owner = parents.get(owner)
            attributes = class_attributes.get(id(owner), {}) if owner is not None else {}
            assigned: list[tuple[str, ast.AST]] = []
            attribute_pairs: list[tuple[str, ast.AST]] = []
            _assignments_in(function.body, assigned, attribute_pairs)
            solved = _solve_kinds(assigned, _parameter_kinds(function), attributes)
            if function_names.get(id(function)) != solved:
                function_names[id(function)] = solved
                changed = True
            if owner is None:
                continue
            current = dict(class_attributes.get(id(owner), {}))
            for attr, value in attribute_pairs:
                kind = _attribute_kind(_expression_kind(value, solved, current))
                merged = _merge_kind(current.get(attr), kind)
                current[attr] = merged
            if _is_property(function):
                for value in _return_values(function):
                    kind = _attribute_kind(_expression_kind(value, solved, current))
                    current[function.name] = _merge_kind(current.get(function.name), kind)
            if class_attributes.get(id(owner)) != current:
                class_attributes[id(owner)] = current
                changed = True
        if not changed:
            break

    literals: set[str] = set()
    private = False
    parameterized = False
    unresolved = False
    seen = 0
    for function in functions:
        owner = parents.get(function)
        while owner is not None and not isinstance(owner, ast.ClassDef):
            owner = parents.get(owner)
        attributes = class_attributes.get(id(owner), {}) if owner is not None else {}
        names = function_names.get(id(function), {})
        for node in ast.walk(function):
            if not _database_call(node):
                continue
            inside = parents.get(node)
            while inside is not None and inside is not function:
                if isinstance(inside, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    break
                inside = parents.get(inside)
            if inside is not function:
                continue
            seen += 1
            argument = _call_argument(node)
            kind = "unresolved" if argument is None else _connect_kind(
                _expression_kind(argument, names, attributes), function.name,
            )
            if kind == "private":
                private = True
            elif kind == "parameterized":
                parameterized = True
            elif kind.startswith("literal:"):
                value = kind.split(":", 1)[1]
                if _dotted_name(node.func) == "sqlite3.connect":
                    literals.add("shared_database:" + value)
                else:
                    database = _postgres_name(value)
                    if database:
                        literals.add("shared_database:postgres:" + database)
                    else:
                        unresolved = True
            else:
                unresolved = True
    walked = sum(1 for node in ast.walk(tree) if _database_call(node))
    if walked != seen:
        unresolved = True
    lowered = text.lower()
    if any(token in lowered for token in ("sqlite3.connect", "psycopg.connect", "postgresql://")) and "connect" in calls and walked == 0:
        unresolved = True
    if unresolved:
        return [_resource(
            filename, "shared_database", "shared_database",
            "static source evidence for a database whose target is not proven private",
            wildcard=True,
        )]
    literals.update(_direct_database_literals(tree))
    found = []
    for scope in sorted(literals):
        found.append(_resource(
            filename, "shared_database", scope,
            "static source evidence for database " + scope.split(":", 1)[1],
        ))
    if (private or parameterized) and not literals:
        found.append(_resource(
            filename, "private_database", "private_database",
            "database path is :memory:, a per-test temporary path, or an instance path chosen by the caller",
        ))
    return found


def _precise_resource_evidence(tree: ast.AST, text: str, calls: set[str], filename: str) -> list[dict]:
    """Split a shared-resource mark only when every target in the file is proven independent.

    An unresolved target keeps the broad lock, and that lock still conflicts with every
    narrower target of the same kind. A private target (:memory:, a per-test temporary
    path, an ephemeral port, a pytest monkeypatch) is not a cross-worker lock.
    """
    found: list[dict] = []
    lowered = text.lower()

    found.extend(_database_evidence(tree, text, calls, filename))

    leaks_environment = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(_environ_target(target) for target in node.targets):
            leaks_environment = True
        elif isinstance(node, ast.Delete) and any(_environ_target(target) for target in node.targets):
            leaks_environment = True
        elif isinstance(node, ast.Call) and _dotted_name(node.func) in {
            "os.environ.update", "os.environ.setdefault", "os.environ.pop",
            "os.environ.clear", "os.environ.__setitem__",
        }:
            leaks_environment = True
    if leaks_environment:
        found.append(_resource(
            filename, "global_environment", "global_environment",
            "assigns os.environ; safe across processes, not inside one pytest process",
            exclusive=True,
        ))
    elif any(token in lowered for token in ("monkeypatch.setenv", "monkeypatch.delenv")) and calls.intersection({"setenv", "delenv"}):
        found.append(_resource(
            filename, "pytest_environment", "pytest_environment",
            "pytest monkeypatch restores the environment after the test",
        ))

    origins: set[str] = set()
    external_unresolved = False
    external_calls = {
        "urlopen", "requests.get", "requests.post", "requests.put", "requests.patch",
        "requests.delete", "requests.head", "requests.request", "httpx.get", "httpx.post",
        "httpx.put", "httpx.patch", "httpx.delete", "httpx.head", "httpx.request",
        "socket.create_connection",
    }
    saw_external = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name not in external_calls and not name.endswith(".urlopen"):
            continue
        saw_external = True
        argument = node.args[0] if node.args else None
        if name == "socket.create_connection" and isinstance(argument, ast.Tuple) and argument.elts:
            host = argument.elts[0]
            if isinstance(host, ast.Constant) and isinstance(host.value, str) and host.value:
                origins.add("external_service:tcp://" + host.value)
            else:
                external_unresolved = True
            continue
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            origin = _http_origin(argument.value)
            if origin:
                origins.add("external_service:" + origin)
            else:
                external_unresolved = True
        else:
            external_unresolved = True
    external_needle = any(
        token in lowered for token in ("requests.", "httpx.", "urllib.request", "socket.create_connection")
    )
    if external_needle and calls.intersection({"urlopen", "create_connection"}) and not saw_external:
        external_unresolved = True
    if external_unresolved:
        found.append(_resource(
            filename, "external_service", "external_service",
            "static source evidence for an external service whose target is not constant",
            wildcard=True,
        ))
    else:
        for scope in sorted(origins):
            found.append(_resource(
                filename, "external_service", scope,
                "static source evidence for " + scope.split(":", 1)[1],
            ))

    ports: set[int] = set()
    port_unresolved = False
    saw_bind = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if not (name in {"socket.bind", "bind"} or name.endswith(".bind")):
            continue
        saw_bind = True
        if not node.args or not isinstance(node.args[0], ast.Tuple) or len(node.args[0].elts) < 2:
            port_unresolved = True
            continue
        port = node.args[0].elts[1]
        if isinstance(port, ast.Constant) and isinstance(port.value, int):
            if port.value != 0:
                ports.add(port.value)
        else:
            port_unresolved = True
    if "socket.bind" in lowered and "bind" in calls and not saw_bind:
        port_unresolved = True
    if port_unresolved:
        found.append(_resource(
            filename, "fixed_tcp_port", "fixed_tcp_port",
            "binds a TCP port that is not a literal",
            wildcard=True,
        ))
    else:
        for port in sorted(ports):
            found.append(_resource(
                filename, "fixed_tcp_port", f"fixed_tcp_port:{port}",
                f"binds fixed TCP port {port}",
            ))

    flock_unresolved = False
    flock_paths: set[str] = set()
    saw_flock = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name not in {"fcntl.flock", "msvcrt.locking"} and not name.endswith(".flock"):
            continue
        saw_flock = True
        argument = node.args[0] if node.args else None
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value:
            flock_paths.add("filesystem_lock:" + argument.value)
        elif argument is not None and _private_path(argument):
            continue
        else:
            flock_unresolved = True
    if any(token in lowered for token in ("fcntl.flock", "msvcrt.locking")) and calls.intersection({"flock", "locking"}) and not saw_flock:
        flock_unresolved = True
    if flock_unresolved:
        found.append(_resource(
            filename, "filesystem_lock", "filesystem_lock",
            "locks a file whose path is not constant",
            wildcard=True,
        ))
    else:
        for scope in sorted(flock_paths):
            found.append(_resource(
                filename, "filesystem_lock", scope,
                "locks " + scope.split(":", 1)[1],
            ))

    profiles: set[str] = set()
    profile_unresolved = False
    saw_profile = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _dotted_name(node.func).endswith("launch_persistent_context"):
            continue
        saw_profile = True
        target = None
        for keyword in node.keywords:
            if keyword.arg == "user_data_dir":
                target = keyword.value
        if target is None and node.args:
            target = node.args[0]
        if target is None:
            profile_unresolved = True
            continue
        if isinstance(target, ast.Constant) and isinstance(target.value, str) and target.value:
            profiles.add("browser_profile:" + target.value)
        elif _private_path(target):
            continue
        else:
            profile_unresolved = True
    if (
        ("user_data_dir" in lowered or "user-data-dir" in lowered or "launch_persistent_context" in lowered)
        and not saw_profile
    ):
        profile_unresolved = True
    if profile_unresolved:
        found.append(_resource(
            filename, "browser_profile", "browser_profile",
            "uses a persistent browser profile whose directory is not constant",
            wildcard=True,
        ))
    else:
        for scope in sorted(profiles):
            found.append(_resource(
                filename, "browser_profile", scope,
                "uses persistent browser profile " + scope.split(":", 1)[1],
            ))

    if any(token in lowered for token in ("docker compose", "docker run", "containerd", "podman")) or "docker" in calls:
        found.append(_resource(
            filename, "docker_resource", "docker_resource",
            "uses the worker's own Docker daemon",
            exclusive=True,
        ))
    return found


def _static_evidence_uncached(
    paths: Sequence[str],
    platform_clues: Sequence[str],
) -> tuple[list[dict], list[dict]]:
    evidence = []
    resources = []
    seen = set()
    for filename in paths:
        try:
            stat = Path(filename).stat()
        except OSError:
            stat = None
        if stat is None:
            tree, text = None, ""
        else:
            tree, text = _parse_file(filename, stat.st_mtime_ns, stat.st_size)
        if tree is None:
            evidence.append({
                "rule": "source-unavailable", "file": filename,
                "detail": "source could not be statically analyzed",
            })
            continue
        imports = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
                    if isinstance(node.func.value, ast.Name):
                        calls.add(node.func.value.id + "." + node.func.attr)
                elif isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
        for system, modules in PLATFORM_IMPORTS.items():
            for module in sorted(imports.intersection(modules)):
                key = (system, "platform-import", module, filename)
                if key not in seen:
                    evidence.append({
                        "rule": "platform-import", "platform": system, "file": filename,
                        "detail": f"imports platform-specific module {module}",
                    })
                    seen.add(key)
        windows_tokens = ("ctypes.windll", "winreg", "powershell", "win32api", "WindowsApps")
        linux_tokens = ("/proc/", "/sys/", "/run/systemd", "AF_UNIX", "systemd")
        for system, tokens in (("Windows", windows_tokens), ("Linux", linux_tokens)):
            for token in tokens:
                if token.lower() in text.lower():
                    key = (system, "platform-api", token, filename)
                    if key not in seen:
                        evidence.append({
                            "rule": "platform-api", "platform": system, "file": filename,
                            "detail": f"references platform-specific API or path {token}",
                        })
                        seen.add(key)
        if any("platform.machine" in clue or "os.uname" in clue for clue in platform_clues) or any(
            token in text.lower() for token in ("aarch64", "arm64", "x86_64", "amd64")
        ):
            evidence.append({
                "rule": "architecture-check", "platform": "Linux",
                "architecture": "arm64", "file": filename,
                "detail": "checks or references host architecture; target matrix needs review",
            })
        if "playwright" in imports or any(
            token in text for token in ("sync_playwright", "async_playwright", "launch_persistent_context")
        ):
            evidence.append({
                "rule": "browser-api", "capability": "browser", "file": filename,
                "detail": "uses Playwright or browser APIs",
            })
        if "docker" in imports or any(
            token in text.lower() for token in ("docker compose", "docker run", "containerd", "podman")
        ):
            evidence.append({
                "rule": "container-api", "capability": "docker", "file": filename,
                "detail": "uses Docker/container APIs or commands",
            })
        resources.extend(_precise_resource_evidence(tree, text, calls, filename))
    for clue in platform_clues:
        if clue in {"os.name", "sys.platform", "platform.system", "platform.machine"}:
            evidence.append({
                "rule": "conditional-platform-check", "file": None,
                "detail": f"test source checks {clue}; branch conditions require review",
            })
    unique_resources = {
        (item["resource"], item["file"], item.get("scope")): item for item in resources
    }
    return evidence, list(unique_resources.values())


def _fixture_resource_evidence(fixture_names: Sequence[str], fixture_details: Sequence[Mapping[str, Any]]) -> list[dict]:
    resources = []
    for entry in fixture_details:
        name = entry["fixture"].lower()
        metadata = entry.get("metadata") or {}
        if any(token in name for token in ("browser", "playwright", "chromium", "page", "context")):
            resources.append({
                "resource": "browser_fixture", "file": metadata.get("source_file"),
                "scope": "browser_fixture",
                "detail": f"depends on browser-related fixture {entry['fixture']}",
                "confidence": "INFERRED",
            })
        if any(token in name for token in ("docker", "container", "oci")):
            resources.append({
                "resource": "docker_resource", "file": metadata.get("source_file"),
                "scope": "docker_resource", "batch_exclusive": True,
                "detail": f"depends on container-related fixture {entry['fixture']}",
                "confidence": "INFERRED",
            })
        if metadata.get("scope") in {"module", "package", "session"}:
            resources.append({
                "resource": "shared_fixture_state",
                "file": metadata.get("source_file"),
                "detail": f"uses {metadata['scope']}-scoped fixture {entry['fixture']}; shared-state isolation needs review",
                "confidence": "INFERRED",
            })
    return resources


def _selector_entry(mapping: Mapping[str, Any], test_id: str) -> tuple[Mapping[str, Any] | None, str | None]:
    file_path = test_id.split("::", 1)[0]
    matches = [
        (selector, entry) for selector, entry in mapping.items()
        if selector == test_id or selector == file_path or fnmatch.fnmatchcase(test_id, selector)
    ]
    if not matches:
        return None, None
    matches.sort(key=lambda pair: (len(pair[0]), pair[0]), reverse=True)
    return matches[0][1], matches[0][0]


def classify_evidence(
    record: Mapping[str, Any],
    *,
    manifest: str | Mapping[str, Any] | None = None,
    validation_manifest: str | Mapping[str, Any] | None = None,
    source_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Classify with evidence; only validated explicit requirements are VERIFIED."""
    manifest_value = load_manifest(manifest)
    validation_value = load_validation_manifest(validation_manifest)
    test_id = stable_test_id(record.get("nodeid"))
    fixture_names, fixture_details, fixture_blockers = _fixture_closure(record)
    source_paths = _source_paths(record, fixture_details)
    static_evidence, isolation = _static_evidence(source_paths, record.get("platform_clues", []))
    isolation.extend(_fixture_resource_evidence(sorted(fixture_names), fixture_details))
    explicit_entry, explicit_selector = _selector_entry(manifest_value["tests"], test_id)
    if explicit_entry is None:
        explicit_entry, explicit_selector = _selector_entry(manifest_value["rules"], test_id)
    fixture_dependencies_verified = bool(
        explicit_entry and explicit_entry.get("fixture_dependencies_verified") is True
    )
    blockers = [] if fixture_dependencies_verified else list(fixture_blockers)
    try:
        base = classify_test(record, manifest_value)
    except ClassificationError as error:
        base = None
        blockers.append(str(error))
    classifier_conflict = bool(
        base is not None and base.status == "CONFLICT"
        or base is None and "conflict" in str(blockers).lower()
    )

    marker_metadata = base.as_dict() if base is not None else None
    requirements: dict[str, Any] = {
        "operating_systems": [],
        "architectures": [],
        "features": [],
        "minimum_ram_bytes": 0,
        "docker_architecture": None,
        "oracle_specific": False,
        "cross_platform_os": [],
        "serial_group": None,
    }
    if base is not None and base.status in {"CLASSIFIED", "CONFLICT"}:
        req = marker_metadata["requirements"]
        requirements = {
            "operating_systems": sorted({item["operating_system"] for item in req["allowed_platforms"]}),
            "architectures": sorted({item["architecture"] for item in req["allowed_platforms"]}),
            "features": list(req["required_features"]),
            "minimum_ram_bytes": req["minimum_ram_bytes"],
            "docker_architecture": req["docker_architecture"],
            "oracle_specific": req["oracle_authorized_environment"],
            "cross_platform_os": req["cross_platform_os"],
            "serial_group": req["serial_group"],
            "isolation_group": req["isolation_group"],
            "isolation_verified": req["isolation_verified"],
            "qualification": req["qualification"],
        }
        if base.status == "CONFLICT":
            blockers.append(base.reason or "conflicting marker requirements")
    if explicit_entry:
        for key in ("operating_systems", "architectures", "features", "minimum_ram_bytes",
                    "docker_architecture", "oracle_specific", "cross_platform_os", "serial_group",
                    "isolation_group", "isolation_verified", "qualification"):
            if key in explicit_entry:
                target_key = "features" if key == "features" else key
                requirements[target_key] = explicit_entry[key]
    elif base is None or base.status == "UNCLASSIFIED":
        features = set()
        oses = set()
        for item in static_evidence:
            if item.get("platform") in {"Windows", "Linux"}:
                oses.add(item["platform"])
            if item.get("capability") == "browser":
                features.update(("playwright", "chromium"))
            if item.get("capability") == "docker":
                features.add("docker")
        if oses:
            requirements["operating_systems"] = sorted(oses)
        if features:
            requirements["features"] = sorted(features)
        if "browser_profile" in {item["resource"] for item in isolation}:
            requirements["features"] = sorted(set(requirements["features"]) | {"playwright", "chromium"})
        if "docker_resource" in {item["resource"] for item in isolation}:
            requirements["features"] = sorted(set(requirements["features"]) | {"docker"})
    fixture_features = {
        feature for fixture in fixture_names
        for feature in manifest_value["fixture_capabilities"].get(fixture, [])
    }
    if fixture_features:
        requirements["features"] = sorted(
            set(requirements.get("features", [])) | fixture_features
        )
    inferred_fixture_features = set()
    for fixture in fixture_names:
        lowered = fixture.lower()
        if any(part in lowered for part in ("browser", "playwright", "chromium", "page", "context")):
            inferred_fixture_features.update(("playwright", "chromium"))
        if any(part in lowered for part in ("docker", "container", "oci")):
            inferred_fixture_features.add("docker")
    if inferred_fixture_features:
        requirements["features"] = sorted(
            set(requirements.get("features", [])) | inferred_fixture_features
        )

    conflict_evidence = []
    oses = requirements["operating_systems"]
    architectures = requirements["architectures"]
    if oses and not set(oses) <= {"Windows", "Linux"}:
        conflict_evidence.append("unsupported operating-system requirement")
    if architectures and not set(architectures) <= {"x64", "arm64"}:
        conflict_evidence.append("unsupported architecture requirement")
    if oses and architectures and not any(
        system in oses and architecture in architectures for system, architecture in SUPPORTED
    ):
        conflict_evidence.append("operating-system and architecture requirements do not intersect")
    if conflict_evidence:
        blockers.extend(conflict_evidence)

    validation_entry, validation_selector = _selector_entry(
        validation_value["validations"], test_id,
    )
    targets = validation_entry.get("targets", {}) if validation_entry else {}
    requirements_digest = hashlib.sha256(
        json.dumps(requirements, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    validation_requirements_match = bool(
        validation_entry
        and validation_entry.get("requirements_sha256") == requirements_digest
    )
    required_pairs = {
        (system, architecture)
        for system, architecture in SUPPORTED
        if (not oses or system in oses) and (not architectures or architecture in architectures)
    }
    if requirements["cross_platform_os"]:
        required_pairs = {
            pair for pair in required_pairs if pair[0] in requirements["cross_platform_os"]
        }
    valid_pairs = set()
    validation_evidence = []
    for target, result in targets.items():
        if not isinstance(result, Mapping):
            continue
        parts = target.split("/")
        if len(parts) != 2 or (parts[0], parts[1]) not in SUPPORTED:
            continue
        pair = (parts[0], parts[1])
        passed = (
            validation_requirements_match
            and result.get("result") == "PASS"
            and isinstance(result.get("source_commit_sha"), str)
            and result.get("source_commit_sha") == source_commit_sha
            and isinstance(result.get("run_id"), str) and bool(result.get("run_id"))
            and isinstance(result.get("evidence"), str) and bool(result.get("evidence"))
        )
        if passed:
            valid_pairs.add(pair)
        validation_evidence.append({
            "target": target,
            "result": result.get("result"),
            "current_source_commit": result.get("source_commit_sha") == source_commit_sha,
            "accepted": passed,
        })
    explicitly_verified = bool(
        (explicit_entry and explicit_entry.get("verified") is True)
        or (base is not None and base.status == "CLASSIFIED")
    )
    if conflict_evidence or classifier_conflict:
        level, status = "UNKNOWN", "CONFLICT"
    elif fixture_blockers and not fixture_dependencies_verified:
        level, status = "UNKNOWN", "UNKNOWN"
    elif (
        explicitly_verified and required_pairs and required_pairs <= valid_pairs
        and not blockers
    ):
        level, status = "VERIFIED", "CLASSIFIED"
    elif base is not None and base.status == "CLASSIFIED" or static_evidence or isolation:
        level, status = "INFERRED", "CLASSIFIED"
    else:
        level, status = "UNKNOWN", "UNCLASSIFIED"

    if level == "VERIFIED":
        categories = list(base.categories if base is not None else [])
        if not categories:
            categories = ["PORTABLE"] if required_pairs == set(SUPPORTED) else []
        if requirements["features"]:
            feature_to_category = {
                "playwright": "BROWSER", "chromium": "BROWSER", "docker": "DOCKER",
            }
            categories.extend(
                feature_to_category[feature] for feature in requirements["features"]
                if feature in feature_to_category
            )
        categories = sorted(set(categories))
    elif level == "INFERRED":
        categories = list(base.categories if base is not None else [])
        if oses and set(oses) == {"Windows"}:
            categories.append("WINDOWS_ONLY")
        if oses and set(oses) == {"Linux"}:
            categories.append("LINUX_ONLY")
        if architectures and set(architectures) == {"arm64"}:
            categories.append("ARM64_NATIVE")
        feature_categories = {
            "playwright": "BROWSER", "chromium": "BROWSER", "docker": "DOCKER",
        }
        categories.extend(feature_categories[item] for item in requirements["features"]
                          if item in feature_categories)
        if requirements["oracle_specific"]:
            categories.append("ORACLE_SPECIFIC")
        if requirements["cross_platform_os"]:
            categories.append("CROSS_PLATFORM")
        if requirements["serial_group"] or any(
            item["resource"] in {"fixed_tcp_port", "filesystem_lock", "global_environment"}
            for item in isolation
        ):
            categories.append("SERIAL_ONLY")
        categories = sorted(set(categories))
    else:
        categories = ["UNCLASSIFIED"] if status != "CONFLICT" else ["CONFLICT"]

    test_file = test_id.split("::", 1)[0]
    family = Path(test_file).stem
    return {
        "test_id": test_id,
        "status": status,
        "confidence_name": level,
        "confidence_level": level,
        "categories": categories,
        "requirements": requirements,
        "classification_source": (
            "manifest-rule:" + str(explicit_selector)
            if explicit_selector and explicit_selector in manifest_value["rules"]
            else "manifest:" + str(explicit_selector) if explicit_selector
            else "pytest-markers" if base is not None and base.status in {"CLASSIFIED", "CONFLICT"}
            else "static-rules" if static_evidence or isolation
            else "pytest-collection"
        ),
        "confidence": {"VERIFIED": 1.0, "INFERRED": 0.65, "UNKNOWN": 0.0}[level],
        "test_family": family,
        "fixture_closure": sorted(fixture_names),
        "fixture_evidence": fixture_details,
        "fixture_graph_complete": not fixture_blockers,
        "fixture_graph_blockers": sorted(set(fixture_blockers)),
        "static_evidence": static_evidence,
        "isolation": isolation,
        "conflicts": conflict_evidence,
        "blockers": sorted(set(blockers)),
        "validation": {
            "selector": validation_selector,
            "required_targets": [f"{system}/{architecture}" for system, architecture in sorted(required_pairs)],
            "validated_targets": [f"{system}/{architecture}" for system, architecture in sorted(valid_pairs)],
            "evidence": validation_evidence,
        },
        "requirements_sha256": requirements_digest,
        "source_sha256": record.get("source_sha256"),
        "incremental": record.get("incremental", "changed"),
        "reason": (
            "; ".join(sorted(set(blockers))) if blockers
            else "explicit requirements await matching worker validation evidence"
            if level == "INFERRED"
            else "insufficient verified requirements or source evidence"
            if level == "UNKNOWN"
            else None
        ),
    }


def build_classification_report(
    inventory: Sequence[Mapping[str, Any]],
    *,
    manifest: str | Mapping[str, Any] | None = None,
    validation_manifest: str | Mapping[str, Any] | None = None,
    source_commit_sha: str | None = None,
    baseline_report: Mapping[str, Any] | None = None,
    devices: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    classification_started = monotonic()
    manifest_value = load_manifest(manifest)
    validation_value = load_validation_manifest(validation_manifest)
    input_fingerprint = hashlib.sha256(json.dumps({
        "manifest": manifest_value,
        "validation_manifest": validation_value,
        "source_commit_sha": source_commit_sha,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    classifications = []
    fixture_catalog = dict((baseline_report or {}).get("fixture_catalog", {}))
    baseline_items = {}
    if (
        baseline_report
        and baseline_report.get("classification_inputs_sha256") == input_fingerprint
    ):
        baseline_items = {
            item.get("test_id"): item for item in baseline_report.get("classifications", [])
            if isinstance(item, Mapping)
        }
    inventory_by_id = {}
    for record in inventory:
        test_id = stable_test_id(record.get("nodeid"))
        if test_id in inventory_by_id:
            raise ClassificationError(f"duplicate inventory test ID: {test_id}")
        inventory_by_id[test_id] = record
    for test_id, record in sorted(inventory_by_id.items()):
        prior = baseline_items.get(test_id)
        if (
            record.get("incremental") == "unchanged" and prior
            and prior.get("source_sha256") == record.get("source_sha256")
        ):
            classification = dict(prior)
            classification["incremental"] = "reused"
        else:
            classification = classify_evidence(
                record, manifest=manifest_value, validation_manifest=validation_value,
                source_commit_sha=source_commit_sha,
            )
            if baseline_report is not None:
                classification["incremental"] = "changed"
        fixture_refs = []
        for entry in classification.get("fixture_evidence", []):
            metadata = entry.get("metadata")
            if not isinstance(metadata, Mapping):
                fixture_refs.append(entry)
                continue
            definition_id = str(metadata.get("definition_id") or _digest(metadata))
            context_fingerprint = _digest(metadata.get("dependency_bindings", {}))[:12]
            catalog_id = f"{definition_id}:{context_fingerprint}"
            catalog_metadata = {
                key: value for key, value in metadata.items()
                if key not in {"parameterized", "parameter_index"}
            }
            fixture_catalog.setdefault(catalog_id, {
                "fixture": entry["fixture"],
                "definition_id": definition_id,
                "metadata": catalog_metadata,
            })
            fixture_ref = {
                "fixture": entry["fixture"],
                "fixture_definition_ref": catalog_id,
            }
            if "parameterized" in metadata:
                fixture_ref["parameterized"] = metadata["parameterized"]
            if metadata.get("parameter_index") is not None:
                fixture_ref["parameter_index"] = metadata["parameter_index"]
            fixture_refs.append(fixture_ref)
        classification["fixture_evidence"] = fixture_refs
        classifications.append(classification)

    counts = Counter(item["confidence_level"] for item in classifications)
    if sum(counts.values()) != len(inventory_by_id):
        raise ClassificationError("classification report does not cover the complete inventory")
    by_os = Counter()
    by_capability = Counter()
    by_family = Counter()
    isolation_counts = Counter()
    fixture_blockers = Counter()
    for item in classifications:
        by_family[item["test_family"]] += 1
        fixture_blockers.update(item.get("fixture_graph_blockers", []))
        requirements = item["requirements"]
        for os_name in requirements.get("operating_systems", []):
            by_os[os_name] += 1
        for feature in requirements.get("features", []):
            by_capability[feature] += 1
        for isolation in item["isolation"]:
            isolation_counts[isolation["resource"]] += 1

    worker_devices = [
        device for device in devices
        if device.get("status") == "ONLINE" and device.get("assigned_task") is None
    ]
    schedulable = []
    required_validation = Counter()
    for item in classifications:
        req = item["requirements"]
        if item["confidence_level"] == "VERIFIED":
            eligible = [
                device for device in worker_devices
                if (not req.get("operating_systems")
                    or device.get("capabilities", {}).get("operating_system") in req["operating_systems"])
                and (not req.get("architectures")
                     or device.get("capabilities", {}).get("architecture") in req["architectures"])
                and device.get("capabilities", {}).get("available_ram_bytes", 0)
                    >= req.get("minimum_ram_bytes", 0)
                and all(
                    device.get("capabilities", {}).get({
                        "playwright": "playwright_available",
                        "chromium": "chromium_available",
                        "docker": "docker_available",
                        "pytest": "pytest_available",
                    }.get(feature, ""), False)
                    for feature in req.get("features", [])
                    if feature in {"playwright", "chromium", "docker", "pytest"}
                )
                and all(
                    feature in device.get("capabilities", {}).get("optional_capabilities", [])
                    for feature in req.get("features", [])
                    if feature not in {"playwright", "chromium", "docker", "pytest"}
                )
                and (
                    not req.get("docker_architecture")
                    or device.get("capabilities", {}).get("docker_architecture")
                    == req["docker_architecture"]
                )
                and (
                    not req.get("oracle_specific")
                    or device.get("oracle_environment_authorized") is True
                )
            ]
            if eligible:
                schedulable.append(item["test_id"])
        if item["confidence_level"] != "VERIFIED":
            for pair in item["validation"]["required_targets"]:
                required_validation[pair] += 1

    conflicts = [
        {"test_id": item["test_id"], "evidence": item["conflicts"], "blockers": item["blockers"]}
        for item in classifications if item["status"] == "CONFLICT"
    ]
    blockers = Counter()
    for item in classifications:
        for reason in item["blockers"]:
            blockers[reason] += 1
    baseline_total = 18657
    previous_total = (baseline_report or {}).get("total_collected_tests", baseline_total)
    changed = sum(item["incremental"] != "reused" for item in classifications)
    evidence_cache = _cached_static_evidence.cache_info()
    parse_cache = _parse_file.cache_info()
    evidence_calls = evidence_cache.hits + evidence_cache.misses
    return {
        "schema_version": 1,
        "source_commit_sha": source_commit_sha,
        "classification_inputs_sha256": input_fingerprint,
        "total_collected_tests": len(classifications),
        "phase2_baseline_total": baseline_total,
        "baseline_delta": len(classifications) - baseline_total,
        "previous_report_total": previous_total,
        "incremental": {
            "changed_or_reclassified": changed,
            "reused_from_previous_report": len(classifications) - changed,
        },
        "counts": {level: counts.get(level, 0) for level in LEVELS},
        "percentages": {
            level: round(100 * counts.get(level, 0) / len(classifications), 2)
            if classifications else 0.0
            for level in LEVELS
        },
        "by_operating_system_requirement": dict(sorted(by_os.items())),
        "by_capability_requirement": dict(sorted(by_capability.items())),
        "by_test_family": dict(sorted(by_family.items())),
        "by_isolation_resource": dict(sorted(isolation_counts.items())),
        "fixture_evidence_encoding": "fixture_catalog_references",
        "fixture_catalog": dict(sorted(fixture_catalog.items())),
        "classification_duration_seconds": round(monotonic() - classification_started, 3),
        "static_analysis_cache": {
            "hits": evidence_cache.hits,
            "misses": evidence_cache.misses,
            "hit_rate": round(evidence_cache.hits / evidence_calls, 4) if evidence_calls else 0.0,
            "entries": evidence_cache.currsize,
            "source_parse_hits": parse_cache.hits,
            "source_parse_misses": parse_cache.misses,
        },
        "fixture_graph_completeness": {
            "complete_test_count": sum(
                item.get("fixture_graph_complete", False) for item in classifications
            ),
            "incomplete_test_count": sum(
                not item.get("fixture_graph_complete", False) for item in classifications
            ),
            "blockers": dict(sorted(fixture_blockers.items())),
        },
        "conflicts": conflicts,
        "remaining_blockers": dict(sorted(blockers.items())),
        "registered_online_workers": len(worker_devices),
        "schedulable_test_count": len(set(schedulable)),
        "schedulable_test_ids": sorted(set(schedulable)),
        "required_validation_targets": dict(sorted(required_validation.items())),
        "classification_coverage_is_execution_result": False,
        "execution_results": {"pass": None, "fail": None, "skip": None},
        "recommendations": [
            "Run focused validation on Windows x64, Linux x64, and Linux ARM64 workers.",
            "Add worker evidence to the validation manifest before promoting INFERRED tests.",
            "Review UNKNOWN fixture dependencies, dynamic fixture lookups, and platform branches.",
            "Verify shared-state isolation before enabling concurrent batches.",
        ],
        "classifications": classifications,
    }
