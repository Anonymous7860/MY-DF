from importlib import import_module as _import_module

from .Sample import Sample
from .Sample import SampleType
from .SampleLoader import SampleLoader
from .SampleProcessor import SampleProcessor
from .SampleGeneratorBase import SampleGeneratorBase
from .SampleGeneratorFace import SampleGeneratorFace
from .SampleGeneratorFaceTemporal import SampleGeneratorFaceTemporal
from .SampleGeneratorImage import SampleGeneratorImage
from .SampleGeneratorImageTemporal import SampleGeneratorImageTemporal
from .SampleGeneratorFaceCelebAMaskHQ import SampleGeneratorFaceCelebAMaskHQ
from .SampleGeneratorFaceXSeg import SampleGeneratorFaceXSeg
from .PackedFaceset import PackedFaceset

_LAZY_EXPORTS = {
    'FaceCache': ('.FaceCache', 'FaceCache'),
    'get_or_create_cache': ('.FaceCache', 'get_or_create_cache'),
    'FaceDataset': ('.FaceDataset', 'FaceDataset'),
    'CachedFaceDataset': ('.FaceDataset', 'CachedFaceDataset'),
    'UniformYawSampler': ('.FaceDataset', 'UniformYawSampler'),
    'InfiniteSampler': ('.FaceDataset', 'InfiniteSampler'),
    'InfiniteUniformYawSampler': ('.FaceDataset', 'InfiniteUniformYawSampler'),
    'DataLoaderIterator': ('.FaceDataset', 'DataLoaderIterator'),
    'create_face_dataloader': ('.FaceDataset', 'create_face_dataloader'),
    'create_face_data_iterator': ('.FaceDataset', 'create_face_data_iterator'),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = _import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
