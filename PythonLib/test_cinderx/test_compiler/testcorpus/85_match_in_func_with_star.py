# pyre-ignore-all-errors
def f():
    match (0, 1, 2):
        case [*x]:
            pass
