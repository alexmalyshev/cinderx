# Portions copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

"""Module symbol-table generator"""
from __future__ import annotations

import ast
import os
import sys

from .consts import (
    CO_FUTURE_ANNOTATIONS,
    SC_CELL,
    SC_FREE,
    SC_GLOBAL_EXPLICIT,
    SC_GLOBAL_IMPLICIT,
    SC_LOCAL,
    SC_UNKNOWN,
)
from .misc import mangle
from .visitor import ASTVisitor

if sys.version_info[0] >= 3:
    long = int

MANGLE_LEN = 256

DEF_NORMAL = 1
DEF_COMP_ITER = 2


class Scope:
    is_function_scope = False

    # XXX how much information do I need about each name?
    def __init__(self, name, module, klass=None, lineno=0):
        self.name = name
        self.module = module
        self.lineno = lineno
        self.defs = {}
        self.uses = {}
        self.globals = {}
        self.explicit_globals = {}
        self.nonlocals = {}
        self.params = {}
        self.frees = {}
        self.cells = {}
        self.type_params: set[str] = set()
        self.children = []
        self.parent = None
        self.coroutine = False
        self.comp_iter_target = self.comp_iter_expr = 0
        # nested is true if the class could contain free variables,
        # i.e. if it is nested within another function.
        self.nested = None
        # It's possible to define a scope (class, function) at the nested level,
        # but explicitly mark it as global. Bytecode-wise, this is handled
        # automagically, but we need to generate proper __qualname__ for these.
        self.global_scope = False
        self.generator = False
        self.klass = None
        self.suppress_jit = False
        self.can_see_class_scope = False
        if klass is not None:
            for i in range(len(klass)):
                if klass[i] != "_":
                    self.klass = klass[i:]
                    break

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.name)

    def mangle(self, name):
        if self.klass is None:
            return name
        return mangle(name, self.klass)

    def add_def(self, name, kind=DEF_NORMAL):
        mangled = self.mangle(name)
        self.defs[mangled] = kind | self.defs.get(mangled, 1)

    def add_use(self, name):
        self.uses[self.mangle(name)] = 1

    def add_global(self, name):
        name = self.mangle(name)
        if name in self.uses or name in self.defs:
            pass  # XXX warn about global following def/use
        if name in self.params:
            raise SyntaxError("{} in {} is global and parameter".format(name, self.name))
        self.explicit_globals[name] = 1
        self.module.add_def(name)
        # Seems to be behavior of Py3.5, "global foo" sets foo as
        # explicit global for module too
        self.module.explicit_globals[name] = 1

    def add_param(self, name):
        name = self.mangle(name)
        self.defs[name] = 1
        self.params[name] = 1

    def add_type_param(self, name: str):
        if name in self.type_params:
            raise SyntaxError("duplicated type parameter: {!r}".format(name))
        self.type_params.add(name)

    def get_names(self):
        d = {}
        d.update(self.defs)
        d.update(self.uses)
        d.update(self.globals)
        return d.keys()

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def get_children(self):
        return self.children

    def inline_nested_comprehensions(self) -> None:
        pass

    def DEBUG(self):
        print(self.name, self.nested and "nested" or "")
        print("\tglobals: ", self.globals)
        print("\texplicit_globals: ", self.explicit_globals)
        print("\tcells: ", self.cells)
        print("\tdefs: ", self.defs)
        print("\tuses: ", self.uses)
        print("\tfrees:", self.frees)

    def check_name(self, name):
        """Return scope of name.

        The scope of a name could be LOCAL, GLOBAL, FREE, or CELL.
        """
        if name in self.explicit_globals:
            return SC_GLOBAL_EXPLICIT
        if name in self.globals:
            return SC_GLOBAL_IMPLICIT
        if name in self.cells:
            return SC_CELL
        if name in self.frees:
            return SC_FREE
        if name in self.defs:
            return SC_LOCAL
        if self.nested and name in self.uses:
            return SC_FREE
        if self.nested:
            return SC_UNKNOWN
        else:
            return SC_GLOBAL_IMPLICIT

    def get_free_vars(self):
        if not self.nested:
            # If we're not nested we can't possibly have any free variables,
            # as we can't close over class variables.  The exception to this
            # rule is __class__ or __classdict__, which we indeed can close over.
            assert set(self.frees).issubset({"__class__", "__classdict__"})
            return list(self.frees)

        free = {}
        free.update(self.frees)
        for name in self.uses.keys():
            if (
                name not in self.defs
                and name not in self.globals
                and name not in self.explicit_globals
            ):
                free[name] = 1
        return sorted(free.keys())

    def handle_children(self):
        for child in self.children:
            child.handle_children()

        self.inline_nested_comprehensions()

        for child in self.children:
            if child.name in self.explicit_globals:
                child.global_scope = True
            if child.nested:
                frees = child.get_free_vars()

                globals = self.add_frees(frees)
                for name in globals:
                    child.force_global(name)
            else:
                for special in ("__class__", "__classdict__"):
                    if special in child.frees:
                        self.add_frees([special])
                    elif special in child.uses and special not in child.defs:
                        child.frees[special] = 1
                        self.add_frees([special])

    def force_global(self, name):
        """Force name to be global in scope.

        Some child of the current node had a free reference to name.
        When the child was processed, it was labelled a free
        variable.  Now that all its enclosing scope have been
        processed, the name is known to be a global or builtin.  So
        walk back down the child chain and set the name to be global
        rather than free.

        Be careful to stop if a child does not think the name is
        free.
        """
        self.globals[name] = 1
        if name in self.frees:
            del self.frees[name]
        # special case for __class__:
        # in a class scope, __class__ is free when used, but defined
        # for its children
        for child in self.children:
            if child.check_name(name) == SC_FREE:
                if not isinstance(self, ClassScope) or name != "__class__":
                    child.force_global(name)

    def add_frees(self, names):
        """Process list of free vars from nested scope.

        Returns a list of names that are either 1) declared global in the
        parent or 2) undefined in a top-level parent.  In either case,
        the nested scope should treat them as globals.
        """
        child_globals = []
        for name in names:
            sc = self.check_name(name)
            if name == "__class__" or name == "__classdict__":
                if isinstance(self, ClassScope) or sc == SC_LOCAL:
                    self.cells[name] = 1
                    continue
                elif self.findParentClass() is not None:
                    self.frees[name] = 1
                    continue
            if self.nested:
                if sc == SC_UNKNOWN or sc == SC_FREE or isinstance(self, ClassScope):
                    self.frees[name] = 1
                elif sc == SC_GLOBAL_IMPLICIT:
                    child_globals.append(name)
                elif isinstance(self, FUNCTION_LIKE_SCOPES) and sc == SC_LOCAL:
                    self.cells[name] = 1
                elif sc != SC_CELL:
                    child_globals.append(name)
            else:
                if sc == SC_LOCAL:
                    self.cells[name] = 1
                elif sc != SC_CELL:
                    child_globals.append(name)
        return child_globals

    def get_cell_vars(self):
        return sorted(self.cells.keys())

    def findParentClass(self):
        parent = self.parent
        while not isinstance(parent, ClassScope):
            if parent is None:
                break
            parent = parent.parent
        return parent


class ModuleScope(Scope):
    __super_init = Scope.__init__

    def __init__(self):
        # Set lineno to 0 so it sorted guaranteedly before any other scope
        self.__super_init("global", self, lineno=0)


class FunctionScope(Scope):
    is_function_scope = True


class GenExprScope(FunctionScope):
    is_function_scope = False

    __counter = 1

    def __init__(self, name, module, klass=None, lineno=0):
        self.__counter += 1
        super().__init__(name, module, klass, lineno)
        self.add_param(".0")


class LambdaScope(FunctionScope):
    __counter = 1

    def __init__(self, module, klass=None, lineno=0):
        self.__counter += 1
        super().__init__("<lambda>", module, klass, lineno=lineno)


class ClassScope(Scope):
    __super_init = Scope.__init__

    def __init__(self, name, module, lineno=0):
        self.__super_init(name, module, name, lineno=lineno)


class TypeParamScope(Scope):
    pass

class TypeAliasScope(Scope):
    pass

class TypeVarBoundScope(Scope):
    pass

class TypeAliasScope(Scope):
    pass

FUNCTION_LIKE_SCOPES = (FunctionScope, TypeVarBoundScope, TypeParamScope, TypeAliasScope)
    
class SymbolVisitor(ASTVisitor):
    _FunctionScope = FunctionScope
    _GenExprScope = GenExprScope
    _LambdaScope = LambdaScope

    def __init__(self, future_flags: int):
        super().__init__()
        self.future_annotations = future_flags & CO_FUTURE_ANNOTATIONS
        self.scopes: dict[ast.AST, Scope] = {}
        self.klass = None
        self.module = ModuleScope()

    # node that define new scopes

    def visitModule(self, node):
        scope = self.module = self.scopes[node] = self.module
        self.visit(node.body, scope)
        for child in scope.children:
            child.handle_children()

    def visitInteractive(self, node):
        scope = self.module = self.scopes[node] = self.module
        self.visit(node.body, scope)

    visitExpression = visitModule

    # pyre-ignore[11]: Pyre doesn't know TypeAlias
    def enter_type_params(self, node: ast.ClassDef|ast.FunctionDef|ast.TypeAlias|ast.AsyncFunctionDef, parent: Scope):
        scope = TypeParamScope(node.name, self.module, self.klass, lineno=node.lineno)
        parent.add_child(scope)
        scope.parent = parent
        # type_params is a list, which is not hashable, so we key off the first element as
        # there is always at least one.
        self.scopes[node.type_params[0]] = scope
        if isinstance(parent, ClassScope):
            scope.can_see_class_scope = True
            scope.add_use("__classdict__")
            parent.add_def("__classdict__")

        if isinstance(node, ast.ClassDef):
            scope.add_def(".type_params")
            scope.add_use(".type_params")
            scope.add_def(".generic_base")
            scope.add_use(".generic_base")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.add_def(".defaults")
            scope.add_param(".defaults")
            if node.args.kw_defaults:
                scope.add_def(".kwdefaults")
        return scope

    # pyre-ignore[11]: Pyre doesn't know TypeVar
    def visitTypeVar(self, node: ast.TypeVar, parent: Scope):
        parent.add_def(node.name)
        parent.add_type_param(node.name)

        if node.bound:
            is_in_class = parent.can_see_class_scope
            scope = TypeVarBoundScope(node.name, self.module)
            scope.parent = parent
            scope.nested = True
            scope.can_see_class_scope = is_in_class
            if is_in_class:
                scope.add_def("__classdict__")

            self.visit(node.bound, scope)
            parent.add_child(scope)

    # pyre-ignore[11]: Pyre doesn't know TypeVarTuple
    def visitTypeVarTuple(self, node: ast.TypeVarTuple, parent: Scope) -> None:
        parent.add_def(node.name)
        parent.add_type_param(node.name)

    # pyre-ignore[11]: Pyre doesn't know ParamSpec
    def visitParamSpec(self, node: ast.ParamSpec, parent: Scope) -> None: 
        parent.add_def(node.name)
        parent.add_type_param(node.name)
        
    def visitFunctionDef(self, node, parent):
        if node.decorator_list:
            self.visit(node.decorator_list, parent)
        parent.add_def(node.name)

        type_params = getattr(node, "type_params", ())
        if type_params:
            parent = self.enter_type_params(node, parent)
            for param in type_params:
                self.visit(param, parent)

        scope = self._FunctionScope(
            node.name, self.module, self.klass, lineno=node.lineno
        )
        scope.coroutine = isinstance(node, ast.AsyncFunctionDef)
        scope.parent = parent
        if parent.nested or isinstance(parent, FUNCTION_LIKE_SCOPES):
            scope.nested = 1
        self.scopes[node] = scope
        self._do_args(scope, node.args)
        if node.returns:
            if not self.future_annotations:
                self.visit(node.returns, parent)
        self.visit(node.body, scope)

        parent.add_child(scope)

    visitAsyncFunctionDef = visitFunctionDef

    _scope_names = {
        ast.GeneratorExp: "<genexpr>",
        ast.ListComp: "<listcomp>",
        ast.DictComp: "<dictcomp>",
        ast.SetComp: "<setcomp>",
    }

    def visitAwait(self, node, scope):
        scope.coroutine = True
        self.visit(node.value, scope)

    def visitGeneratorExp(self, node, parent):
        scope = self._GenExprScope(
            self._scope_names[type(node)],
            self.module,
            self.klass,
            lineno=node.lineno,
        )
        scope.parent = parent

        # bpo-37757: For now, disallow *all* assignment expressions in the
        # outermost iterator expression of a comprehension, even those inside
        # a nested comprehension or a lambda expression.
        scope.comp_iter_expr = parent.comp_iter_expr
        if isinstance(node, ast.GeneratorExp):
            scope.generator = True

        if (
            parent.nested
            or isinstance(parent, FUNCTION_LIKE_SCOPES)
            or isinstance(parent, GenExprScope)
        ):
            scope.nested = 1

        parent.comp_iter_expr += 1
        self.visit(node.generators[0].iter, parent)
        parent.comp_iter_expr -= 1

        self.visitcomprehension(node.generators[0], scope, True)

        for comp in node.generators[1:]:
            self.visit(comp, scope, False)

        if isinstance(node, ast.DictComp):
            self.visit(node.value, scope)
            self.visit(node.key, scope)
        else:
            self.visit(node.elt, scope)

        self.scopes[node] = scope

        parent.add_child(scope)

    # Whether to generate code for comprehensions inline or as nested scope
    # is configurable, but we compute nested scopes for them unconditionally
    # TODO: this may be not correct, check.
    visitSetComp = visitGeneratorExp
    visitListComp = visitGeneratorExp
    visitDictComp = visitGeneratorExp

    def visitcomprehension(self, node, scope, is_outmost):
        if node.is_async:
            scope.coroutine = True

        scope.comp_iter_target = 1
        self.visit(node.target, scope)
        scope.comp_iter_target = 0
        if is_outmost:
            scope.add_use(".0")
        else:
            scope.comp_iter_expr += 1
            self.visit(node.iter, scope)
            scope.comp_iter_expr -= 1
        for if_ in node.ifs:
            self.visit(if_, scope)

    def visitGenExprInner(self, node, scope):
        for genfor in node.quals:
            self.visit(genfor, scope)

        self.visit(node.expr, scope)

    def visitGenExprFor(self, node, scope):
        self.visit(node.assign, scope)
        self.visit(node.iter, scope)
        for if_ in node.ifs:
            self.visit(if_, scope)

    def visitGenExprIf(self, node, scope):
        self.visit(node.test, scope)

    def visitLambda(self, node, parent):
        scope = self._LambdaScope(self.module, self.klass, lineno=node.lineno)
        scope.parent = parent
        # bpo-37757: For now, disallow *all* assignment expressions in the
        # outermost iterator expression of a comprehension, even those inside
        # a nested comprehension or a lambda expression.
        scope.comp_iter_expr = parent.comp_iter_expr
        if parent.nested or isinstance(parent, FUNCTION_LIKE_SCOPES):
            scope.nested = 1
        self.scopes[node] = scope
        self._do_args(scope, node.args)
        self.visit(node.body, scope)
        parent.add_child(scope)

    def _do_args(self, scope, args):
        for n in args.defaults:
            self.visit(n, scope.parent)
        for n in args.kw_defaults:
            if n:
                self.visit(n, scope.parent)

        for arg in args.posonlyargs:
            name = arg.arg
            scope.add_param(name)
            if arg.annotation and not self.future_annotations:
                self.visit(arg.annotation, scope.parent)
        for arg in args.args:
            name = arg.arg
            scope.add_param(name)
            if arg.annotation and not self.future_annotations:
                self.visit(arg.annotation, scope.parent)
        for arg in args.kwonlyargs:
            name = arg.arg
            scope.add_param(name)
            if arg.annotation and not self.future_annotations:
                self.visit(arg.annotation, scope.parent)
        if args.vararg:
            scope.add_param(args.vararg.arg)
            if args.vararg.annotation and not self.future_annotations:
                self.visit(args.vararg.annotation, scope.parent)
        if args.kwarg:
            scope.add_param(args.kwarg.arg)
            if args.kwarg.annotation and not self.future_annotations:
                self.visit(args.kwarg.annotation, scope.parent)

    def visitClassDef(self, node: ast.ClassDef, parent):
        if node.decorator_list:
            self.visit(node.decorator_list, parent)

        parent.add_def(node.name)

        type_params = getattr(node, "type_params", ())

        orig_parent = parent
        if type_params:
            parent = self.enter_type_params(node, parent)
            for param in type_params:
                self.visit(param, parent)

        for kw in node.keywords:
            self.visit(kw.value, parent)

        for n in node.bases:
            self.visit(n, parent)
        scope = ClassScope(node.name, self.module, lineno=node.lineno)
        # Set parent ASAP. TODO: Probably makes sense to do that for
        # other scope types either.
        scope.parent = parent
        if type_params:
            scope.add_def("__type_params__")
            scope.add_use(".type_params")
                    
        if parent.nested or isinstance(parent, FUNCTION_LIKE_SCOPES):
            scope.nested = 1
        doc = ast.get_docstring(node, False)
        if doc is not None:
            scope.add_def("__doc__")
        scope.add_def("__module__")
        scope.add_def("__qualname__")
        self.scopes[node] = scope
        prev = self.klass
        self.klass = node.name
        self.visit(node.body, scope)
        self.klass = prev
        parent.add_child(scope)

    def visitTypeAlias(self, node: ast.TypeAlias, parent):
        self.visit(node.name, parent)
        
        in_class = isinstance(parent, ClassScope)
        is_generic = len(node.type_params) > 0
        if is_generic:
            parent = self.enter_type_params(node, parent)
            self.visit(node.type_params, parent)

        scope = TypeAliasScope(node.name.id, self.module)
        scope.parent = parent
        scope.can_see_class_scope = in_class
        if in_class:
            scope.add_use("__classdict__")
            scope.parent.add_def("__classdict__")

        self.scopes[node] = scope
        self.visit(node.value, scope)
        parent.add_child(scope)

    # name can be a def or a use
    def visitName(self, node, scope):
        if isinstance(node.ctx, ast.Store):
            if scope.comp_iter_target:
                # This name is an iteration variable in a comprehension,
                # so check for a binding conflict with any named expressions.
                # Otherwise, mark it as an iteration variable so subsequent
                # named expressions can check for conflicts.
                if node.id in scope.nonlocals or node.id in scope.globals:
                    raise SyntaxError(
                        f"comprehension inner loop cannot rebind assignment expression target '{node.id}'"
                    )
                scope.add_def(node.id, DEF_COMP_ITER)

            scope.add_def(node.id)
        elif isinstance(node.ctx, ast.Del):
            # We do something to var, so even if we "undefine" it, it's a def.
            # Implementation-wise, delete is storing special value (usually
            # NULL) to var.
            scope.add_def(node.id)
        else:
            scope.add_use(node.id)

            if node.id == "super" and isinstance(scope, FUNCTION_LIKE_SCOPES):
                # If super() is used, and special cell var __class__ to class
                # definition, and free var to the method. This is complicated
                # by the fact that original Python2 implementation supports
                # free->cell var relationship only if free var is defined in
                # a scope marked as "nested", which normal method in a class
                # isn't.
                scope.add_use("__class__")

    # operations that bind new names

    def visitMatchAs(self, node, scope):
        if node.pattern:
            self.visit(node.pattern, scope)
        if node.name:
            scope.add_def(node.name)

    def visitMatchStar(self, node, scope):
        if node.name:
            scope.add_def(node.name)

    def visitMatchMapping(self, node, scope):
        self.visit(node.keys, scope)
        self.visit(node.patterns, scope)
        if node.rest:
            scope.add_def(node.rest)

    def visitNamedExpr(self, node, scope):
        if scope.comp_iter_expr:
            # Assignment isn't allowed in a comprehension iterable expression
            raise SyntaxError(
                "assignment expression cannot be used in a comprehension iterable expression"
            )

        name = node.target.id
        if isinstance(scope, GenExprScope):
            cur = scope
            while cur:
                if isinstance(cur, GenExprScope):
                    if cur.defs.get(name, 0) & DEF_COMP_ITER:
                        raise SyntaxError(
                            f"assignment expression cannot rebind comprehension iteration variable '{name}'"
                        )

                elif isinstance(cur, FunctionScope):
                    # If we find a FunctionBlock entry, add as GLOBAL/LOCAL or NONLOCAL/LOCAL
                    scope.frees[name] = 1
                    if name not in cur.explicit_globals:
                        scope.nonlocals[name] = 1
                    else:
                        scope.add_use(name)
                    cur.add_def(name)
                    break
                elif isinstance(cur, ModuleScope):
                    scope.globals[name] = 1
                    scope.add_use(name)
                    cur.add_def(name)
                    break
                elif isinstance(cur, ClassScope):
                    raise SyntaxError(
                        "assignment expression within a comprehension cannot be used in a class body"
                    )
                cur = cur.parent

        self.visit(node.value, scope)
        self.visit(node.target, scope)

    def visitFor(self, node, scope):
        self.visit(node.target, scope)
        self.visit(node.iter, scope)
        self.visit(node.body, scope)
        if node.orelse:
            self.visit(node.orelse, scope)

    visitAsyncFor = visitFor

    def visitImportFrom(self, node, scope):
        for alias in node.names:
            if alias.name == "*":
                continue
            scope.add_def(alias.asname or alias.name)

    def visitImport(self, node, scope):
        for alias in node.names:
            name = alias.name
            i = name.find(".")
            if i > -1:
                name = name[:i]
            scope.add_def(alias.asname or name)

    def visitGlobal(self, node, scope):
        for name in node.names:
            scope.add_global(name)

    def visitNonlocal(self, node, scope):
        # TODO: Check that var exists in outer scope
        for name in node.names:
            scope.frees[name] = 1
            scope.nonlocals[name] = 1

    def visitAssign(self, node, scope):
        """Propagate assignment flag down to child nodes.

        The Assign node doesn't itself contains the variables being
        assigned to.  Instead, the children in node.nodes are visited
        with the assign flag set to true.  When the names occur in
        those nodes, they are marked as defs.

        Some names that occur in an assignment target are not bound by
        the assignment, e.g. a name occurring inside a slice.  The
        visitor handles these nodes specially; they do not propagate
        the assign flag to their children.
        """
        for n in node.targets:
            self.visit(n, scope)
        self.visit(node.value, scope)

    def visitAnnAssign(self, node, scope):
        target = node.target
        if isinstance(target, ast.Name):
            if not isinstance(scope, ModuleScope) and (
                target.id in scope.nonlocals or target.id in scope.explicit_globals
            ):
                is_nonlocal = target.id in scope.nonlocals
                raise SyntaxError(
                    f"annotated name '{target.id}' can't be {'nonlocal' if is_nonlocal else 'global'}"
                )
            if node.simple or node.value:
                scope.add_def(target.id)
        else:
            self.visit(node.target, scope)
        if not self.future_annotations:
            self.visit(node.annotation, scope)
        if node.value:
            self.visit(node.value, scope)

    def visitAssName(self, node, scope):
        scope.add_def(node.name)

    def visitAssAttr(self, node, scope):
        self.visit(node.expr, scope)

    def visitSubscript(self, node, scope):
        self.visit(node.value, scope)
        self.visit(node.slice, scope)

    def visitAttribute(self, node, scope):
        self.visit(node.value, scope)

    def visitSlice(self, node, scope):
        if node.lower:
            self.visit(node.lower, scope)
        if node.upper:
            self.visit(node.upper, scope)
        if node.step:
            self.visit(node.step, scope)

    def visitAugAssign(self, node, scope):
        # If the LHS is a name, then this counts as assignment.
        # Otherwise, it's just use.
        self.visit(node.target, scope)
        if isinstance(node.target, ast.Name):
            self.visit(node.target, scope)
        self.visit(node.value, scope)

    # prune if statements if tests are false

    _const_types = str, bytes, int, long, float

    def visitIf(self, node, scope):
        self.visit(node.test, scope)
        self.visit(node.body, scope)
        if node.orelse:
            self.visit(node.orelse, scope)

    # a yield statement signals a generator

    def visitYield(self, node, scope):
        scope.generator = True
        if node.value:
            self.visit(node.value, scope)

    def visitYieldFrom(self, node, scope):
        scope.generator = True
        if node.value:
            self.visit(node.value, scope)

    def visitTry(self, node, scope):
        self.visit(node.body, scope)
        # Handle exception capturing vars
        for handler in node.handlers:
            if handler.type:
                self.visit(handler.type, scope)
            if handler.name:
                scope.add_def(handler.name)
            self.visit(handler.body, scope)
        self.visit(node.orelse, scope)
        self.visit(node.finalbody, scope)


class CinderFunctionScope(FunctionScope):
    def __init__(self, name, module, klass=None, lineno=0):
        super().__init__(name=name, module=module, klass=klass, lineno=lineno)
        self._inlinable_comprehensions = []
        self._inline_comprehensions = os.getenv("PYTHONINLINECOMPREHENSIONS")

    def add_comprehension(self, comp):
        if self._inline_comprehensions:
            self._inlinable_comprehensions.append(comp)

    def inline_nested_comprehensions(self) -> None:
        if not self._inlinable_comprehensions:
            return
        # collect set of names that should not be shadowed
        # by new names introduced by comprehensions
        local_names = set(self.defs.keys()) | self.uses.keys()

        for child in self.children:
            # include all free/implicitly global names from children
            for free in child.get_free_vars():
                sc = self.check_name(free)
                if sc == SC_FREE or sc == SC_GLOBAL_IMPLICIT:
                    local_names.add(free)

        if ".0" in local_names:
            local_names.remove(".0")

        for comp in self._inlinable_comprehensions:
            # unsafe to inline a comprehension with cells
            if comp.cells:
                continue
            # this check is only necessary to match symtable.c, where it's hard
            # to distinguish "has cells" from "otherwise has child scopes with
            # free vars"
            if any(c.get_free_vars() for c in comp.children):
                continue

            # do not inline comprehensions if new names would
            # conflict with existing local names
            # exclude non-locals as they are defined in outer scope
            defs = set(comp.defs.keys()) - comp.nonlocals.keys()
            if defs & local_names:
                continue

            # merge defs from comprehension scope into current scope
            for v in defs:
                if v != ".0":
                    self.add_def(v)

            # for names that are free in comprehension
            # and not present in defs of current scope -
            # add them as free in current scope
            for d in comp.uses:
                if comp.check_name(d) == SC_FREE and d not in self.defs:
                    sc = self.check_name(d)
                    if sc == SC_UNKNOWN:
                        # name is missing in current scope - add it
                        self.frees[d] = 1

            # go through free names in comprehension
            # and check if current scope has corresponding def
            # if yes - name is no longer free after inlining
            for f in list(comp.frees.keys()):
                if f in self.defs:
                    del comp.frees[f]

            # move names uses in comprehension to current scope
            for u in comp.uses.keys():
                if u != ".0":
                    self.add_use(u)

            # cell vars in comprehension become cells in current scope
            for c in comp.cells.keys():
                if c != ".0":
                    self.cells[c] = 1

            # splice children of comprehension into current scope
            # replacing existing entry for 'comp'
            i = self.children.index(comp)
            self.children[i : i + 1] = comp.children
            for c in comp.children:
                c.parent = self

            # mark comprehension as inlined
            comp.inlined = True


class CinderGenExprScope(GenExprScope, CinderFunctionScope):
    inlined = False


class CinderLambdaScope(LambdaScope, CinderFunctionScope):

    pass


class CinderSymbolVisitor(SymbolVisitor):
    _FunctionScope = CinderFunctionScope
    _GenExprScope = CinderGenExprScope
    _LambdaScope = CinderLambdaScope

    def visitGeneratorExp(self, node, parent):
        scope = self._GenExprScope(
            self._scope_names[type(node)],
            self.module,
            self.klass,
            lineno=node.lineno,
        )
        scope.parent = parent

        # bpo-37757: For now, disallow *all* assignment expressions in the
        # outermost iterator expression of a comprehension, even those inside
        # a nested comprehension or a lambda expression.
        scope.comp_iter_expr = parent.comp_iter_expr
        if isinstance(node, ast.GeneratorExp):
            scope.generator = True
        elif isinstance(parent, FunctionScope):
            # record itself as possibly inlinable comprehension in parent scope
            parent.add_comprehension(scope)

        if (
            parent.nested
            or isinstance(parent, FunctionScope)
            or isinstance(parent, GenExprScope)
        ):
            scope.nested = 1

        parent.comp_iter_expr += 1
        self.visit(node.generators[0].iter, parent)
        parent.comp_iter_expr -= 1

        self.visitcomprehension(node.generators[0], scope, True)

        for comp in node.generators[1:]:
            self.visit(comp, scope, False)

        if isinstance(node, ast.DictComp):
            self.visit(node.value, scope)
            self.visit(node.key, scope)
        else:
            self.visit(node.elt, scope)

        self.scopes[node] = scope

        parent.add_child(scope)

    visitSetComp = visitGeneratorExp
    visitListComp = visitGeneratorExp
    visitDictComp = visitGeneratorExp

    def visitLambda(self, node, parent):
        scope = self._LambdaScope(self.module, self.klass, lineno=node.lineno)
        scope.parent = parent
        # bpo-37757: For now, disallow *all* assignment expressions in the
        # outermost iterator expression of a comprehension, even those inside
        # a nested comprehension or a lambda expression.
        scope.comp_iter_expr = parent.comp_iter_expr
        if parent.nested or isinstance(parent, FunctionScope):
            scope.nested = 1
        self.scopes[node] = scope
        self._do_args(scope, node.args)
        self.visit(node.body, scope)

        parent.add_child(scope)

    def visitFunctionDef(self, node, parent):
        if node.decorator_list:
            self.visit(node.decorator_list, parent)
        parent.add_def(node.name)
        scope = self._FunctionScope(
            node.name, self.module, self.klass, lineno=node.lineno
        )
        scope.coroutine = isinstance(node, ast.AsyncFunctionDef)
        scope.parent = parent
        if parent.nested or isinstance(parent, FunctionScope):
            scope.nested = 1
        self.scopes[node] = scope
        self._do_args(scope, node.args)
        if node.returns:
            if not self.future_annotations:
                self.visit(node.returns, parent)
        self.visit(node.body, scope)

        parent.add_child(scope)

    visitAsyncFunctionDef = visitFunctionDef


def list_eq(l1, l2):
    return sorted(l1) == sorted(l2)
