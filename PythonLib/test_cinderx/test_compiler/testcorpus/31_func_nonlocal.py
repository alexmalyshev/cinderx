# pyre-ignore-all-errors
def foo():
    a = 1

    def bar():
        nonlocal a
        a = 2
