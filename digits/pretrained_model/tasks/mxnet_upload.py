# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask


@subclass
class MxnetUploadTask(UploadPretrainedModelTask):

    def __init__(self,** kwargs):
        super(MxnetUploadTask,self).__init__(**kwargs)

    @override
    def name(self):
        return 'Upload Pretrained Mxnet Model'

    @override
    def get_model_def_path(self):
        print self.job_dir
        print '^^^^^^^^^^^^^^^^^^^^^^'
        return os.path.join(self.job_dir,"model.py")

    @override
    def get_weights_path(self):
        return  os.path.join(self.job_dir," s")

    @override
    def __setstate__(self, state):
        super(MxnetUploadTask,self).__setstate__(state)

    @override
    def run(self, resources):

        print '----------------------------->'
        print self.weights_path
        print self.get_weights_path
        print '----------------------------->'
        '''
        self.move_file(self.weights_path, "_Model.t7")
        self.move_file(self.model_def_path, "original.lua")

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt")

        self.status = Status.DONE
        '''