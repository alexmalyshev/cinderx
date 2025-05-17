#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-unsafe

import os
import os.path
import pathlib
import subprocess

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def find_files(path: str, pred: Callable[[str], bool]) -> list[str]:
    result = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if pred(f):
                # Files are returned relative to the top directory.
                abs_file = os.path.join(root, f)
                result.append(os.path.relpath(abs_file, path))
    return result


def find_native_sources(path: str) -> list[str]:
    return find_files(path, lambda f: f.endswith(".cpp") or f.endswith(".c"))


def find_python_sources(path: str) -> list[str]:
    return find_files(path, lambda f: f.endswith(".py"))


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        # Stop the base class from building any sources.
        super().__init__(name, sources=[])


class CMakeBuildExt(build_ext):
    def run(self) -> None:
        for extension in self.extensions:
            self._run_cmake(extension)
        super().run()

    def _run_cmake(self, extension: Extension) -> None:
        cwd = os.path.abspath(os.getcwd())

        build_dir = self.build_temp
        os.makedirs(build_dir, exist_ok=True)

        extension_dir = os.path.abspath(self.get_ext_fullpath(extension.name))
        os.makedirs(extension_dir, exist_ok=True)

        cc_result = subprocess.run(
            ["which", "clang"], capture_output=True, encoding="utf-8"
        )
        if cc_result.returncode != 0:
            raise RuntimeError("Cannot find `clang` binary")
        cc = cc_result.stdout.strip()

        cxx_result = subprocess.run(
            ["which", "clang++"], capture_output=True, encoding="utf-8"
        )
        if cxx_result.returncode != 0:
            raise RuntimeError("Cannot find `clang++` binary")
        cxx = cxx_result.stdout.strip()

        build_type = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.dirname(extension_dir)}",
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        ]

        build_args = [
            "--config",
            build_type,
            "--",
            "-j",
            "8",
        ]

        self.spawn(["cmake"] + cmake_args + ["-B", build_dir, cwd])
        self.spawn(["cmake", "--build", build_dir] + build_args)

    def _find_binary(self, name: str) -> str:
        pass


def main() -> None:
    project_dir = os.path.dirname(__file__)

    # Native sources.
    native_sources = find_native_sources(project_dir)

    # Python sources.
    python_sources = find_python_sources(os.path.join(project_dir, "PythonLib"))
    python_sources = [f for f in python_sources if not f.startswith("test_cinderx/")]

    sources = native_sources + python_sources

    setup(
        name="cinderx",
        version="0.1",
        ext_modules=[CMakeExtension(".")],
        sources=sources,
        cmdclass={
            "build_ext": CMakeBuildExt,
        },
#        packages=find_packages("PythonLib"),
        package_dir = {
            "cinderx": "PythonLib/cinderx",
        },
        python_requires="==3.12",
    )


if __name__ == "__main__":
    main()
