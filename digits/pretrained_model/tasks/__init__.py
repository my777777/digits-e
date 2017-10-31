# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .upload_pretrained import UploadPretrainedModelTask
from .caffe_upload import CaffeUploadTask
from .torch_upload import TorchUploadTask
from .mxnet_upload import MxnetUploadTask

__all__ = [
    'UploadPretrainedModelTask',
    'CaffeUploadTask',
    'TorchUploadTask',
    'MxnetUploadTask',
]
