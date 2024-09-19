# pyre-strict
import contextlib
import functools
import json
import os
import re
import subprocess
from typing import Generator

from clang.cindex import Config, Index, TranslationUnit, TranslationUnitLoadError

CompilationCommand = dict[str, str | list[str]]
CompilationDb = list[CompilationCommand]


@functools.cache
def get_fbsource_root() -> str:
    # This is a hack to make sure we are exactly in the root of an fbsource
    # checkout. This is needed because the compile commands database as
    # generated by buck has directories relatively to that location, and
    # setting the library path for LLVM. This is a bit dirty because it breaks
    # things like RE. Should work well enough for the cases we care about.
    try:
        fbsource_root = subprocess.run(
            ["hg", "root"], capture_output=True, encoding="utf-8", check=True
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception(
            f"Failing stdout:\n{e.stdout}\nFailing stderr:\n{e.stderr}\n"
        ) from e
    return fbsource_root


def filter_cpp_args(args: list[str]) -> list[str]:
    new_args = []
    append_next = False
    for arg in args:
        if append_next:
            new_args.append(arg)
            append_next = False
        elif arg in ["-I", "-isystem", "-idirafter", "-iquote", "-D", "-U", "-Xclang"]:
            new_args.append(arg)
            append_next = True
        elif (
            arg
            in [
                "-nostdinc",
                "-nostdinc++",
                "-no-canonical-prefixes",
                "-pthread",
                "-no-pthread",
                "-pthreads",
            ]
            or arg.startswith("-finput-charset=")
            or arg.startswith("-I")
            or arg.startswith("-U")
            or arg.startswith("-D")
            or arg.startswith("-std=")
        ):
            new_args.append(arg)
    return new_args


class ParsedFile:
    source_file: str
    translation_unit: TranslationUnit

    def __init__(self, command: CompilationCommand) -> None:
        source_file = command["file"]
        assert type(source_file) is str
        print("Loading: ", source_file)
        self.source_file = source_file
        with chdir(get_fbsource_root()):
            args = command["arguments"]
            assert type(args) is list
            args = filter_cpp_args(args)
            self.translation_unit = self.parse_file(source_file, args)
            with open(source_file, "r") as f:
                self.file_content: str = f.read()

    @classmethod
    def from_db(
        cls, source_file: str, compile_commands_db: CompilationDb
    ) -> "ParsedFile":
        for command in compile_commands_db:
            file = command["file"]
            assert type(file) is str
            if file.endswith(source_file):
                return cls(command)
        raise Exception(f"Could not find compile command for {source_file}")

    def parse_file(self, source_file: str, args: list[str]) -> TranslationUnit:
        index = Index.create()
        try:
            return index.parse(source_file, args=args)
        except TranslationUnitLoadError as e:
            # Unfortunately this is the main exception raised from index.parse()
            # and is completely useless. Chances are the error is in the args or
            # the file path.
            raise Exception(
                f"Failed to parse file: {source_file}, used args: {args}"
            ) from e


@contextlib.contextmanager
def chdir(path: str) -> Generator[None, None, None]:
    d = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(d)


def make_compilation_db(compile_commands: str) -> CompilationDb:
    with open(compile_commands, "r") as f:
        compile_commands_db: CompilationDb = json.load(f)

    llvm_driver_example = compile_commands_db[0]["arguments"][1]
    m = re.match(
        r"--cc=fbcode/third-party-buck/platform(\d+)/build/llvm-fb/(\d+)/",
        llvm_driver_example,
    )
    if not m:
        raise Exception(f"Could not find LLVM version in {llvm_driver_example}")
    platform_version = m.group(1)
    llvm_version = int(m.group(2))
    # LLVM prior to 17 (specifically 15) seems to have a bug which slightly
    # breaks parsing of some files.
    llvm_version = max(llvm_version, 17)
    llvm_lib_path = f"fbcode/third-party-buck/platform{platform_version}/build/llvm-fb/{llvm_version}/lib"
    print(f"Setting LLVM library path to: {llvm_lib_path}")
    Config.set_library_path(llvm_lib_path)
    return compile_commands_db


class FileParser:
    def __init__(self, compile_commands: str) -> None:
        self.compilation_db: CompilationDb = make_compilation_db(compile_commands)
        self.parsed_files: dict[str, ParsedFile] = {}

    def parse(self, source_file: str) -> ParsedFile:
        if self.parsed_files.get(source_file) is None:
            self.parsed_files[source_file] = ParsedFile.from_db(
                source_file, self.compilation_db
            )
        return self.parsed_files[source_file]
