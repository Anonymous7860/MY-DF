from importlib import import_module as _import_module

from .FaceType import FaceType


_LAZY_EXPORTS = {
    'LandmarksProcessor': ('.LandmarksProcessor', None),
    'S3FDExtractor': ('.S3FDExtractor', 'S3FDExtractor'),
    'FANExtractor': ('.FANExtractor', 'FANExtractor'),
    'SAENXTExtractor': ('.SAENXTExtractor', 'SAENXTExtractor'),
    'FaceEnhancer': ('.FaceEnhancer', 'FaceEnhancer'),
    'XSegNet': ('.XSegNet', 'XSegNet'),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = _import_module(module_name, __name__)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value
