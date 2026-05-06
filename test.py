import time

import cinderx.jit
cinderx.jit.auto()


def unicode_eq_const(a: str) -> bool:
    return a == "def"


def unicode_eq(a: str, b: str) -> bool:
    return a == b


def ns_to_str(ns: int) -> str:
    if ns <= 1_000:
        return f"{ns}ns"
    elif ns <= 1_000_000:
        return f"{ns // 1_000}us"
    return f"{ns // 1_000_000}ms"


start = time.perf_counter_ns()

# Test both length comparison early exits and full byte comparisons.
for i in range(10000000):
    unicode_eq_const("abc")
    unicode_eq_const("wxyz")

    unicode_eq("abc", "def")
    unicode_eq("tuv", "wxyz")

end = time.perf_counter_ns()
elapsed = end - start

print(f"Completed in {ns_to_str(elapsed)}")
