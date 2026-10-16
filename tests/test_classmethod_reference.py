import ast
import contextvars
import io
from pathlib import Path
import importlib
import pickle
import subprocess
import sys
import types

import pytest

import cloudpickle

_CONTEXT = contextvars.ContextVar("classmethod_reference", default="default")


class ReferenceBase:
    @classmethod
    def read_context(cls):
        return cls.__name__, _CONTEXT.get()


class ReferenceChild(ReferenceBase):
    pass


class ShadowedReferenceChild(ReferenceBase):
    read_context = staticmethod(ReferenceBase.read_context.__func__)


class CustomMeta(type):
    pass


class CustomMetaClass(metaclass=CustomMeta):
    @classmethod
    def read_context(cls):
        return _CONTEXT.get()


class ValueClass:
    @classmethod
    def value(cls):
        return cls.__name__, 42


class CallableMethod:
    def __call__(self, cls):
        return cls.__name__, "callable"


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_class_bound_callable_without_function_name(protocol):
    method = types.MethodType(CallableMethod(), ValueClass)
    clone = pickle.loads(cloudpickle.dumps(method, protocol))
    assert clone.__self__ is ValueClass
    assert clone() == ("ValueClass", "callable")


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("cls", [ReferenceBase, ReferenceChild])
def test_importable_classmethod_reference(cls, protocol):
    method = cls.read_context
    # The standard pickler is a control: the ContextVar need not be serialized.
    assert pickle.loads(pickle.dumps(method, protocol)) == method
    token = _CONTEXT.set("current")
    try:
        clone = pickle.loads(cloudpickle.dumps(method, protocol))
        assert clone.__self__ is cls
        assert clone.__func__ is method.__func__
        assert clone() == (cls.__name__, "current")
    finally:
        _CONTEXT.reset(token)


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_importable_classmethod_fresh_process(protocol):
    payload = cloudpickle.dumps(ReferenceChild.read_context, protocol)
    script = """
import pickle
import sys
from tests.test_classmethod_reference import ReferenceChild, _CONTEXT
_CONTEXT.set('child-process')
method = pickle.loads(sys.stdin.buffer.read())
assert method.__self__ is ReferenceChild
assert method.__func__ is ReferenceChild.read_context.__func__
assert method() == ('ReferenceChild', 'child-process')
"""
    subprocess.run([sys.executable, "-c", script], input=payload, check=True)


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_importable_classmethod_uses_current_attribute(monkeypatch, protocol):
    standard_payload = pickle.dumps(ValueClass.value, protocol)
    payload = cloudpickle.dumps(ValueClass.value, protocol)
    monkeypatch.setattr(ValueClass, "value", classmethod(lambda cls: "changed"))
    assert pickle.loads(standard_payload)() == "changed"
    assert pickle.loads(payload)() == "changed"


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_classmethod_registered_by_value(protocol):
    module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(module)
    try:
        payload = cloudpickle.dumps(ValueClass.value, protocol)
    finally:
        cloudpickle.unregister_pickle_by_value(module)
    script = """
import pickle
import sys
from tests.test_classmethod_reference import ValueClass
ValueClass.value = classmethod(lambda cls: ('changed', 0))
method = pickle.loads(sys.stdin.buffer.read())
assert method() == ('ValueClass', 42)
"""
    subprocess.run([sys.executable, "-c", script], input=payload, check=True)


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_inherited_classmethod_base_registered_by_value(
    tmp_path, monkeypatch, protocol
):
    (tmp_path / "classmethod_reference_base.py").write_text(
        "class Base:\n"
        "    @classmethod\n"
        "    def value(cls):\n"
        "        return 'original'\n"
    )
    (tmp_path / "classmethod_reference_child.py").write_text(
        "from classmethod_reference_base import Base\n"
        "class Child(Base):\n"
        "    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    base_module = importlib.import_module("classmethod_reference_base")
    child_module = importlib.import_module("classmethod_reference_child")
    cloudpickle.register_pickle_by_value(base_module)
    try:
        payload = cloudpickle.dumps(child_module.Child.value, protocol)
        monkeypatch.setattr(
            base_module.Base, "value", classmethod(lambda cls: "changed")
        )
        method = pickle.loads(payload)
        assert method.__self__ is child_module.Child
        assert method() == "original"
    finally:
        cloudpickle.unregister_pickle_by_value(base_module)
        sys.modules.pop("classmethod_reference_base", None)
        sys.modules.pop("classmethod_reference_child", None)


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_detached_class_bound_method(protocol):
    def value(cls):
        return cls.__name__, "detached"

    # Same function name as an existing classmethod, but not that descriptor.
    method = types.MethodType(value, ValueClass)
    clone = pickle.loads(cloudpickle.dumps(method, protocol))
    assert clone.__self__ is ValueClass
    assert clone() == ("ValueClass", "detached")


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_shadowed_inherited_classmethod(protocol):
    # The source-defined function is referenced independently of its binding.
    method = types.MethodType(
        ReferenceBase.read_context.__func__, ShadowedReferenceChild
    )
    clone = pickle.loads(cloudpickle.dumps(method, protocol))
    assert clone.__self__ is ShadowedReferenceChild
    assert clone.__func__ is ReferenceBase.read_context.__func__


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_custom_metaclass_classmethod_keeps_existing_reduction(protocol):
    # Custom metaclasses can change attribute lookup; leave them untouched.
    with pytest.raises(TypeError, match="ContextVar"):
        cloudpickle.dumps(CustomMetaClass.read_context, protocol)


@pytest.mark.parametrize("protocol", [2, pickle.HIGHEST_PROTOCOL])
def test_dynamic_classmethod_contextvar_stays_unsupported(protocol):
    class Dynamic:
        @classmethod
        def value(cls):
            return _CONTEXT.get()

    with pytest.raises(TypeError, match="ContextVar"):
        cloudpickle.dumps(Dynamic.value, protocol)


class RuntimeOriginal:
    @classmethod
    def value(cls):
        return "disk-original"


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_runtime_installed_local_classmethod(monkeypatch, protocol):
    closed = "runtime-closure"

    def value(cls):
        return closed

    monkeypatch.setattr(RuntimeOriginal, "value", classmethod(value))
    payload = cloudpickle.dumps(RuntimeOriginal.value, protocol)
    script = """
import pickle
import sys
assert pickle.loads(sys.stdin.buffer.read())() == 'runtime-closure'
"""
    subprocess.run([sys.executable, "-c", script], input=payload, check=True)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("reverse", [False, True])
def test_classmethod_shared_function_graph(protocol, reverse):
    method = ValueClass.value
    objects = [method, method.__func__]
    if reverse:
        objects.reverse()
    clone = pickle.loads(cloudpickle.dumps(objects, protocol))
    if reverse:
        clone.reverse()
    assert clone[0].__func__ is clone[1]


class ProbeChild(RuntimeOriginal):
    pass


def probe_plain(cls):
    return "plain"


class ProbeWrappedPlain:
    plain = classmethod(probe_plain)


class ProbeAliased:
    alias = RuntimeOriginal.value


class ProbeCustom(metaclass=CustomMeta):
    @classmethod
    def value(cls):
        return "custom"


class ProbeDescriptor(classmethod):
    pass


class ProbeDescriptorClass:
    @ProbeDescriptor
    def value(cls):
        return "descriptor"


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize(
    "method",
    [
        RuntimeOriginal.value,
        ProbeChild.value,
        ProbeAliased.alias,
        ProbeWrappedPlain.plain,
        ProbeCustom.value,
        ProbeDescriptorClass.value,
        types.MethodType(CallableMethod(), RuntimeOriginal),
    ],
)
def test_reviewer_controls(method, protocol):
    clone = pickle.loads(cloudpickle.dumps(method, protocol))
    assert clone() == method()
    assert clone.__self__ is method.__self__


@pytest.fixture
def pure_python_branch_pickler():
    # AST branch harness, NOT execution on an actual PyPy runtime. Compile only
    # the existing Pickler class, selecting its fallback branch and pure-Python
    # base. No reducer bodies are rewritten and no reviewer files are imported.
    import cloudpickle.cloudpickle as implementation

    tree = ast.parse(Path(implementation.__file__).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Pickler"
    )
    branches = [
        node
        for node in cls.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "PYPY"
    ]
    assert len(branches) == 1
    branches[0].test = ast.Constant(value=False)
    namespace = dict(vars(implementation))
    namespace["pickle"] = types.SimpleNamespace(
        **dict(vars(pickle), Pickler=pickle._Pickler)
    )
    module = ast.Module(body=[cls], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), implementation.__file__, "exec"),
        namespace,
    )
    return namespace["Pickler"]


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize(
    "case", ["method", "function", "graph", "custom_meta", "custom_descriptor"]
)
def test_pure_python_branch_harness(pure_python_branch_pickler, protocol, case):
    method = ValueClass.value
    objects = {
        "method": method,
        "function": method.__func__,
        "graph": [method, method.__func__],
        "custom_meta": ProbeCustom.value,
        "custom_descriptor": ProbeDescriptorClass.value,
    }
    stream = io.BytesIO()
    pure_python_branch_pickler(stream, protocol).dump(objects[case])
    clone = pickle.loads(stream.getvalue())
    if case == "function":
        assert clone(ValueClass) == method.__func__(ValueClass)
    elif case == "graph":
        assert clone[0].__func__ is clone[1]
        assert clone[0]() == method()
    else:
        assert clone() == objects[case]()
