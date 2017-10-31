# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import operator
import os
import re
import shutil
import subprocess
import tempfile
import time
import sys
import h5py
import numpy as np
import PIL.Image

from .train import TrainTask
import digits
from digits import utils
from digits.config import config_value
from digits.utils import subclass, override, constants

# Must import after importing digit.config


# NOTE: Increment this every time the pickled object changes
PICKLE_VERSION = 1

# Constants
MXNET_MODEL_FILE = 'model.py'
MXNET_SNAPSHOT_PREFIX = 'snapshot'


######################################
def ip_match(str):
    x = ''
    if str == '192.168.0.1':
        x = '19'
    if str == '192.168.0.2':
        x = '17'
    if str == '192.168.0.3':
        x = '15'
    if str == '192.168.0.4':
        x = '13'
    if str == '192.168.0.5':
        x = '11'
    if str == '192.168.0.6':
        x = '9'
    return x



############################################

def subprocess_visible_devices(gpus):
    """
    Calculates CUDA_VISIBLE_DEVICES for a subprocess
    """
    if not isinstance(gpus, list):
        raise ValueError('gpus should be a list')
    gpus = [int(g) for g in gpus]

    old_cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if old_cvd is None:
        real_gpus = gpus
    else:
        map_visible_to_real = {}
        for visible, real in enumerate(old_cvd.split(',')):
            map_visible_to_real[visible] = int(real)
        real_gpus = []
        for visible_gpu in gpus:
            real_gpus.append(map_visible_to_real[visible_gpu])
    return ','.join(str(g) for g in real_gpus)


@subclass
class MxnetTrainTask(TrainTask):
    """
    Trains a mxnet model
    """

    MXNET_LOG = 'mxnet_output.log'

    def __init__(self, **kwargs):
        """
        Arguments:
        network -- a NetParameter defining the network
        """
        super(MxnetTrainTask, self).__init__(**kwargs)

        # save network description to file

        with open(os.path.join(self.job_dir, MXNET_MODEL_FILE), "w") as outfile:
            outfile.write(self.network)

        self.pickver_task_mxnet_train = PICKLE_VERSION

        self.current_epoch = 0
        self.current_iteration = 0
        ####################################################170716
        self.first_procee = True

        ####################################################
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None
        self.image_mean = None
        self.classifier = None
        self.solver = None

        self.model_file = MXNET_MODEL_FILE
        self.snapshot_prefix = MXNET_SNAPSHOT_PREFIX
        self.log_file = self.MXNET_LOG

        self.digits_version = digits.__version__

    def __getstate__(self):
        state = super(MxnetTrainTask, self).__getstate__()

        # Don't pickle these things
        if 'labels' in state:
            del state['labels']
        if 'image_mean' in state:
            del state['image_mean']
        if 'classifier' in state:
            del state['classifier']
        if 'mxnet_log' in state:
            del state['mxnet_log']

        return state

    def __setstate__(self, state):
        super(MxnetTrainTask, self).__setstate__(state)

        # Make changes to self
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None

        # These things don't get pickled
        self.image_mean = None
        self.classifier = None

    # Task overrides

    @override
    def name(self):
        return 'Train Mxnet Model'

    @override
    def before_run(self):
        super(MxnetTrainTask, self).before_run()
        self.mxnet_log = open(self.path(self.MXNET_LOG), 'a')
        self.saving_snapshot = False
        self.receiving_train_output = False
        self.receiving_val_output = False
        self.last_train_update = None
        self.displaying_network = False
        return True

    def create_mean_file(self):
        filename = os.path.join(self.job_dir, constants.MEAN_FILE_IMAGE)

        return filename

    @override
    def task_arguments(self, resources, env):

        stdmxnetexehome = '/home/mayang/mxnet-0.11.0/example/image-classification/'
        ygymxnetexehome = '/home/mayang/mxnet/example/image-classification/'


        # print self.networkName, "~~~~",self.dataset.name()
        # print '~~~~~~~~~~~~~~~~~~~'
        AlgorithmOpt= self.AlgorithmOpt
        machine_select= self.machine_select
        # algorithmOpt =1:Async-SGD,  =2:Sync-SGD,  =3:Grouping-SGD
        # machine select = 1 means:single machine
        # machine select = 2 means:multi machine
        # print '<><><><><><><>@@@@@'
        datasetname = self.dataset.name()
        if datasetname=='mnist':
            mxnetexefiles = 'train_mnist.py'
        elif datasetname =='cifar10':
            mxnetexefiles ='train_cifar10.py'
        else:
            mxnetexefiles ='train_mnist.py'


        if(int(AlgorithmOpt)==1):
            mxnetexefiles = 'asp_'+datasetname+'_'+self.networkName+'.sh '
        if (int(AlgorithmOpt) == 2):
            mxnetexefiles = 'bsp_' + datasetname + '_' + self.networkName + '.sh '
        if (int(AlgorithmOpt) == 3):
            mxnetexefiles = 'gsp_' + datasetname + '_' + self.networkName + '.sh '

        # print mxnetexefiles
        # print '0-----)>>>>>>'

        ####################################################################################
        if ((int(AlgorithmOpt) == 1) or (int(AlgorithmOpt) == 2) or (int(AlgorithmOpt) == 3)):
            if self.hostlist:
                ob = open('/home/mayang/mxnet/example/image-classification/hosts', 'w')
                hosts = self.hostlist.split(',')
                for i in range(len(hosts)):
                    ob.write(hosts[i] + '\n')

        if (int(AlgorithmOpt) == 3):

            class dataDistribution(object):
                def __init__(self, No, Ip, speed, perc_b, perc_e, nodeNum):
                    self.No = No
                    self.Ip = Ip
                    self.speed = speed
                    self.perc_b = perc_b
                    self.perc_e = perc_e
                    self.nodeNum = nodeNum
            IdenNode = []
            IdenIp = []
            ob2 = open('/home/mayang/mxnet/example/image-classification/clusting/NumIP.txt', 'r')
            while 1:
                line = ob2.readline()
                if line:
                    lline = line.split(',')
                    IdenNode.append(lline[0].strip())
                    IdenIp.append(lline[1].strip())
                else:
                    break
            ob2.close()

            ob5 = open('/home/mayang/mxnet/example/image-classification/clusting/node_speed.txt')
            node_speed = []
            while 1:
                line = ob5.readline()
                if line:
                    lline = line.split(',')
                    if lline[0].strip() in IdenIp:
                        node_speed.append(float(lline[1].strip()))
                else:
                    break
            #
            # manualgrouping = '[1,4][2,3]'

            ob3 = open('/home/mayang/mxnet/example/image-classification/group.txt', 'w')
            ob4 = open('/home/mayang/mxnet/example/image-classification/hosts', 'w')
            groupInfo = self.manualgrouping.split('][')
            s = []
            dist = []
            checkSum = 0
            t = 0
            for i in range(len(groupInfo)):
                xlist = groupInfo[i].strip('[] ').split(',')
                for j in range(len(xlist)):
                    check = 1000
                    index = IdenNode.index(xlist[j].strip())
                    # varIp = ip_match(IdenIp[index])
                    s.append(xlist[j].strip())
                    # ob3.write(varIp + ' ')
                    if check > node_speed[index]:
                        check = node_speed[index]

                    _d = dataDistribution(xlist[j].strip(), IdenIp[index], check, 0, 0, len(xlist))
                    dist.append(_d)
                    t = t + 1
                ob3.write('\n')
                for j in range(t - len(xlist), t):
                    dist[j].speed = check
                checkSum = checkSum + check * len(xlist)

            s.sort()
            for i in range(len(s)):
                ob4.write(IdenIp[int(s[i]) - 1] + '\n')
            import subprocess
            for i in range(len(IdenIp)):
                subprocess.Popen("scp /home/mayang/mxnet/example/image-classification/group.txt " + IdenIp[
                    i] + ":/home/mayang/mxnet/example/image-classification/", shell=True)

            for i in range(len(dist)):
                if i == len(dist) - dist[len(dist) - 1].nodeNum:
                    if i == 0:
                        inter = round(1.00 / dist[len(dist) - 1].nodeNum, 2)
                        for j in range(dist[i].nodeNum):
                            if j == 0:
                                dist[i + j].perc_b = 0
                            else:
                                dist[i + j].perc_b = dist[i + j - 1].perc_e
                            dist[i + j].perc_e = dist[i + j].perc_b + inter
                    else:
                        inter = round((1 - dist[i - 1].perc_e) / dist[len(dist) - 1].nodeNum, 2)
                        for j in range(dist[i].nodeNum):
                            dist[i + j].perc_b = dist[i + j - 1].perc_e
                            dist[i + j].perc_e = dist[i + j].perc_b + inter
                    break
                elif i > 0 and i < (len(dist) - dist[len(dist) - 1].nodeNum):
                    dist[i].perc_b = dist[i - 1].perc_e
                    dist[i].perc_e = dist[i].perc_b + round((dist[i].speed) / checkSum, 2)
                else:
                    dist[i].perc_e = dist[i].perc_b + round((dist[i].speed) / checkSum, 2)



            import operator
            cmpfun = operator.attrgetter('No')
            dist.sort(key=cmpfun)
            dataDistribution = ''
            for i in range(len(dist)):
                dataDistribution = dataDistribution + ' --cluster' + str(dist[i].No) + '-begin ' + str(
                    dist[i].perc_b) + ' --cluster' + str(dist[i].No) + '-end ' + str(dist[i].perc_e)



        ######################################################################

        if(int(self.machine_select)==1):
            args = [sys.executable, stdmxnetexehome + mxnetexefiles]
        else:
            args = [ygymxnetexehome + mxnetexefiles]



        # mxnetexefiles = 'train_mnist.py'
        # args = [sys.executable, stdmxnetexehome + mxnetexefiles]


        if (int(machine_select)==1):
            if 'gpus' in resources:
                identifiers = []
                for identifier, value in resources['gpus']:
                    identifiers.append(identifier)
                if len(identifiers) == 1:
                    args.append('--gpu=%s' % identifiers[0])
                elif len(identifiers) > 1:
                    args.append('--gpus=%s' % ','.join(identifiers))
        #print args


        if (int(machine_select) == 1):
            lr = self.learning_rate
            train_epochs = self.train_epochs #num_epochs
            networkName = self.networkName
            batch_size=self.batch_size

            args.append('--lr=%s' % self.learning_rate)
            args.append('--network=%s' % self.networkName)
            if (int(self.machine_select) == 1):
                args.append('--num-epochs=%d' % int(self.train_epochs) )
            else:
                args.append('--num-epoch=%d' % int(self.train_epochs))


            if self.batch_size is not None:
                args.append('--batch-size=%d' % self.batch_size)
            else:
                if (int(self.machine_select) == 2):
                    args.append('--batch-size=25 ')
            print args
            print '---------------'
        else:
            args.append(' %f ' % self.learning_rate)
            args.append(' %s ' % self.networkName)
            args.append(' %d ' % int(self.train_epochs))

            if self.batch_size is not None:
                args.append(' %d ' % self.batch_size)
            else:
                args.append(' 25 ')

            if (int(AlgorithmOpt) == 3):
                args.append(dataDistribution)
                ##########128/jiedianshu quzheng
                # print dataDistribution

            print args
            print '---------------'
        # assert (0)
        return args



    @override
    def process_output(self, line):

        self.mxnet_log.write('%s\n' % line)
        self.mxnet_log.flush()
        # print line
        # parse mxnet output
        timestamp, level, message = self.preprocess_output_mxnet(line)


        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            print level
            print 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
            self.p.send_signal(15)
            sigterm_time = time.time()

            from digits.status import Status, StatusCls
            self.status = Status.ABORT
            return False

        ###################################################################
        if not message:
            return True
        #   1=0.962082  shu zi  = shu zi

        """
        [0] Batch [900]	Speed: 7244.69 samples/sec	accuracy=0.958281
        [0] Train-accuracy=0.948480
        [0] Time cost=7.467
        [1] Validation-accuracy=0.962082
        """
        ########################################################170716
        if self.current_epoch == 0 and self.first_procee:
            self.save_train_output('TrainAcc','Accuracy', 0.00)
            self.save_val_output('accuracy','Accuracy', 0.00)
            self.first_procee = False
            if (int(self.machine_select) == 1):
                self.save_train_output('learning_rate', 'LearningRate', self.learning_rate)
        ###############################################################

        print message

        # test accurancy
        float_exp = '(NaN|[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)'
        match = re.match(r'\[(\d+)\] Validation-accuracy=(%s)' % float_exp, message)
        if match:
            ################## dan ji
            epoch = match.group(1)
        #match = re.match(r'(\d+) (%s)' % float_exp,message,flags=re.IGNORECASE)
        #if match:
            i = int(epoch)+1########################################
            print 'send progress update signl~~~',i
            self.new_iteration(i)
            value = match.group(2)
            self.save_val_output('accuracy','Accuracy',float(value))




        ###################################################
        match = re.match(r'\[(\d+)\] Batch \[100\].*accuracy=(%s)' % float_exp,message)
        if match:
            ################### danji
            i = int(match.group(1)) + 1  ########################################
            print 'send progress update signl~~~', i
            self.new_iteration(i)
            trainvalue = match.group(2)
            self.save_train_output('TrainAcc','Accuracy',float(trainvalue))
            self.save_train_output('learning_rate', 'LearningRate',float(self.learning_rate))
        ###################################################################
        match = re.match(r'Epoch\[(\d+)\] Validation-accuracy=(%s)' % float_exp, message)
        if match:
            epoch = match.group(1)
            i = int(epoch)+1
            self.new_iteration(i)
            value = match.group(2)
            self.save_val_output('accuracy', 'Accuracy', float(value))

        match = re.match(r'Epoch\[(\d+)\] Train-accuracy=(%s)' % float_exp, message)
        if match:
            epoch = match.group(1)
            i = int(epoch) + 1
            self.new_iteration(i)
            value = match.group(2)
            self.save_train_output('TrainAcc','Accuracy', float(value))
        # net output
        #match = re.match(r'(Train|Test) net output #(\d+): (\S*) = %s' % float_exp, message, flags=re.IGNORECASE)
            #############Server[2] Update[18001]: Change learning rate to 5.72995e-04
            #############match = re.match(r'.*(Server\[2\] Update\[\d+\]: Change learning rate to .*)',line)
        # match = re.match(r'Server\[0\] Update\[\d+\]: Change learning rate to (%s)' % float_exp, message)
        # if match:
        #
        #     value = match.group(1)
        #     self.save_train_output('learning_rate', 'LearningRate',  float(value))
        ###################################################################
        return True

    def new_iteration(self, it):
        """
        Update current_iteration
        """
        if ((self.current_iteration == it) and (it!=0)):
            return
        self.current_iteration = it


        #print 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
        #self.current_epoch =it
        self.send_progress_update(float(it))
        #print '())()()()()()()()()()'
        #print it
        #print float(it)/5

        #print '())()()()()()()()()()'

    @staticmethod
    def preprocess_output_mxnet(line):
        """
        Takes line of output and parses it according to caffe's output format
        Returns (timestamp, level, message) or (None, None, None)
        """
        # NOTE: This must change when the logging format changes
        # LMMDD HH:MM:SS.MICROS pid file:lineno] message
        '''
2017-07-26 22:51:35,728 Server[3] Update[18001]: Change learning rate to 5.72995e-04
2017-07-26 22:51:19,436 Server[1] Update[18001]: Change learning rate to 5.72995e-04
2017-07-26 22:51:19,437 Server[2] Update[18001]: Change learning rate to 5.72995e-04
2017-07-26 22:51:19,436 Server[0] Update[18001]: Change learning rate to 5.72995e-04
2017-07-26 22:51:35,730 Server[5] Update[18001]: Change learning rate to 5.72995e-04

        '''
        match = re.match(r'.*(Server\[0\] Update\[\d+\]: Change learning rate to .*)',line)
        if match:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            level = 'info'
            message = match.group(1)

            print message

            return (timestamp,level,message)
        """
        INFO:root:Epoch[0] Batch [900]	Speed: 7244.69 samples/sec	accuracy=0.958281
        INFO:root:Epoch[0] Train-accuracy=0.948480
        INFO:root:Epoch[0] Time cost=7.467
        INFO:root:Epoch[1] Validation-accuracy=0.962082
        """
        # print '------------------****'
        # print line
        # print '------------------****'

        # match = re.match(r'(\w*):root:Epoch\[(\d+)\] Validation-accuracy=(%s)' % float_exp, line)
        print line
        match = re.match(r'CalledProcessError:.*',line)
        if match:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            level = 'critical'
            message = 'CalledProcessError'
            print level
            return (timestamp, level, message)

        match = re.match(r'.*failed.*', line)
        if match:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            level = 'critical'
            message = 'launch failed'
            print level
            return (timestamp, level, message)

        match = re.match(r'(\w*):root:Epoch(.*)', line)
        if match:
            #time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

            timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            level = match.group(1)
            message = match.group(2)
            # epoch = match.group(2)
            # value = match.group(3)

            # print  epoch,value,timestamp,level
            # print '~~~~~'
            # print '-----curr acc :::', value
            if level == 'INFO':
                level = 'info'
            elif level == 'WARNING':
                level = 'warning'
            elif level == 'ERROR':
                level = 'error'
            elif level == 'FAIL':  # FAIL
                level = 'critical'
            return (timestamp, level, message)
        else:
            match = re.match(r'.*Node\[2\] (Epoch.*)',line)
            if match:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                level = 'info'
                message = match.group(1)
                print message
                print '<><><><><><><>><><><'
                return (timestamp, level, message)
            else:
                return (None, None, None)

    def send_snapshot_update(self):
        """
        Sends socketio message about the snapshot list
        """
        # TODO: move to TrainTask
        from digits.webapp import socketio

        socketio.emit('task update',
                      {
                          'task': self.html_id(),
                          'update': 'snapshots',
                          'data': self.snapshot_list(),
                      },
                      namespace='/jobs',
                      room=self.job_id,
                      )



    # TrainTask overrides
    @override
    def after_run(self):
        super(MxnetTrainTask,self).after_run()
        self.mxnet_log.close()

    @override
    def after_runtime_error(self):
        if os.path.exists(self.path(self.MXNET_LOG)):
            output = subprocess.check_output(['tail', '-n40', self.path(self.MXNET_LOG)])
            lines = []
            for line in output.split('\n'):
                # parse mxnet header
                timestamp, level, message = self.preprocess_output_mxnet(line)

                if message:
                    lines.append(message)
            # return the last 20 lines
            traceback = '\n\nLast output:\n' + '\n'.join(lines[len(lines) - 20:]) if len(lines) > 0 else ''
            if self.traceback:
                self.traceback = self.traceback + traceback
            else:
                self.traceback = traceback

            if 'DIGITS_MODE_TEST' in os.environ:
                print output

    @override
    def detect_snapshots(self):
        self.snapshots = []

        snapshot_dir = os.path.join(self.job_dir, os.path.dirname(self.snapshot_prefix))
        snapshots = []

        for filename in os.listdir(snapshot_dir):
            # find models
            match = re.match(r'%s_(\d+)\.?(\d*)(_Weights|_Model)\.t7' %
                             os.path.basename(self.snapshot_prefix), filename)
            if match:
                epoch = 0
                if match.group(2) == '':
                    epoch = int(match.group(1))
                else:
                    epoch = float(match.group(1) + '.' + match.group(2))
                snapshots.append((
                    os.path.join(snapshot_dir, filename),
                    epoch
                )
                )

        self.snapshots = sorted(snapshots, key=lambda tup: tup[1])

        return len(self.snapshots) > 0

    @override
    def est_next_snapshot(self):
        # TODO: Currently this function is not in use. Probably in future we may have to implement this
        return None

    @override
    def infer_one(self,
                  data,
                  snapshot_epoch=None,
                  layers=None,
                  gpu=None,
                  resize=True):
        # resize parameter is unused
        return self.infer_one_image(data,
                                    snapshot_epoch=snapshot_epoch,
                                    layers=layers,
                                    gpu=gpu)

    def infer_one_image(self, image, snapshot_epoch=None, layers=None, gpu=None):
        """
        Classify an image
        Returns (predictions, visualizations)
            predictions -- an array of [ (label, confidence), ...] for each label, sorted by confidence
            visualizations -- an array of (layer_name, activations, weights) for the specified layers
        Returns (None, None) if something goes wrong

        Arguments:
        image -- a np.array

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        layers -- which layer activation[s] and weight[s] to visualize
        """
        temp_image_handle, temp_image_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_image_handle)
        image = PIL.Image.fromarray(image)
        try:
            image.save(temp_image_path, format='png')
        except KeyError:
            error_message = 'Unable to save file to "%s"' % temp_image_path
            self.logger.error(error_message)
            raise digits.inference.errors.InferenceError(error_message)

        file_to_load = self.get_snapshot(snapshot_epoch)

        args = [config_value('mxnet')['executable'],
                os.path.join(
                    os.path.dirname(os.path.abspath(digits.__file__)),
                    'tools', 'mxnet', 'wrapper.lua'),
                'test.lua',
                '--image=%s' % temp_image_path,
                '--network=%s' % self.model_file.split(".")[0],
                '--networkDirectory=%s' % self.job_dir,
                '--snapshot=%s' % file_to_load,
                '--allPredictions=yes',
                ]
        if hasattr(self.dataset, 'labels_file'):
            args.append('--labels=%s' % self.dataset.path(self.dataset.labels_file))

        if self.use_mean != 'none':
            filename = self.create_mean_file()
            args.append('--mean=%s' % filename)

        if self.use_mean == 'pixel':
            args.append('--subtractMean=pixel')
        elif self.use_mean == 'image':
            args.append('--subtractMean=image')
        else:
            args.append('--subtractMean=none')

        if self.crop_size:
            args.append('--crop=yes')
            args.append('--croplen=%d' % self.crop_size)

        if layers == 'all':
            args.append('--visualization=yes')
            args.append('--save=%s' % self.job_dir)

        # Convert them all to strings
        args = [str(x) for x in args]

        regex = re.compile('\x1b\[[0-9;]*m', re.UNICODE)  # TODO: need to include regular expression for MAC color codes
        self.logger.info('%s classify one task started.' % self.get_framework_id())

        unrecognized_output = []
        predictions = []
        self.visualization_file = None

        env = os.environ.copy()

        if gpu is not None:
            args.append('--type=cuda')
            # make only the selected GPU visible
            env['CUDA_VISIBLE_DEVICES'] = subprocess_visible_devices([gpu])
        else:
            args.append('--type=float')

        p = subprocess.Popen(args,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             cwd=self.job_dir,
                             close_fds=True,
                             env=env,
                             )

        try:
            while p.poll() is None:
                for line in utils.nonblocking_readlines(p.stdout):
                    if self.aborted.is_set():
                        p.terminate()
                        raise digits.inference.errors.InferenceError(
                            '%s classify one task got aborted. error code - %d'
                            % (self.get_framework_id(), p.returncode))

                    if line is not None:
                        # Remove color codes and whitespace
                        line = regex.sub('', line).strip()
                    if line:
                        if not self.process_test_output(line, predictions, 'one'):
                            self.logger.warning('%s classify one task unrecognized input: %s' %
                                                (self.get_framework_id(), line.strip()))
                            unrecognized_output.append(line)
                    else:
                        time.sleep(0.05)

        except Exception as e:
            if p.poll() is None:
                p.terminate()
            error_message = ''
            if type(e) == digits.inference.errors.InferenceError:
                error_message = e.__str__()
            else:
                error_message = '%s classify one task failed with error code %d \n %s' % (
                    self.get_framework_id(), p.returncode, str(e))
            self.logger.error(error_message)
            if unrecognized_output:
                unrecognized_output = '\n'.join(unrecognized_output)
                error_message = error_message + unrecognized_output
            raise digits.inference.errors.InferenceError(error_message)

        finally:
            self.after_test_run(temp_image_path)

        if p.returncode != 0:
            error_message = '%s classify one task failed with error code %d' % (self.get_framework_id(), p.returncode)
            self.logger.error(error_message)
            if unrecognized_output:
                unrecognized_output = '\n'.join(unrecognized_output)
                error_message = error_message + unrecognized_output
            raise digits.inference.errors.InferenceError(error_message)
        else:
            self.logger.info('%s classify one task completed.' % self.get_framework_id())

        predictions = {'output': np.array(predictions)}

        visualizations = []

        if layers == 'all' and self.visualization_file:
            vis_db = h5py.File(self.visualization_file, 'r')
            # the HDF5 database is organized as follows:
            # <root>
            # |- layers
            #    |- 1
            #    |  |- name
            #    |  |- activations
            #    |  |- weights
            #    |- 2
            for layer_id, layer in vis_db['layers'].items():
                layer_desc = layer['name'][...].tostring()
                if 'Sequential' in layer_desc or 'Parallel' in layer_desc:
                    # ignore containers
                    continue
                idx = int(layer_id)
                # activations
                if 'activations' in layer:
                    data = np.array(layer['activations'][...])
                    # skip batch dimension
                    if len(data.shape) > 1 and data.shape[0] == 1:
                        data = data[0]
                    vis = utils.image.get_layer_vis_square(data)
                    mean, std, hist = self.get_layer_statistics(data)
                    visualizations.append(
                        {
                            'id':         idx,
                            'name':       layer_desc,
                            'vis_type':   'Activations',
                            'vis': vis,
                            'data_stats': {
                                'shape':      data.shape,
                                'mean':       mean,
                                'stddev':     std,
                                'histogram':  hist,
                            }
                        }
                    )
                # weights
                if 'weights' in layer:
                    data = np.array(layer['weights'][...])
                    if 'Linear' not in layer_desc:
                        vis = utils.image.get_layer_vis_square(data)
                    else:
                        # Linear (inner product) layers have too many weights
                        # to display
                        vis = None
                    mean, std, hist = self.get_layer_statistics(data)
                    parameter_count = reduce(operator.mul, data.shape, 1)
                    if 'bias' in layer:
                        bias = np.array(layer['bias'][...])
                        parameter_count += reduce(operator.mul, bias.shape, 1)
                    visualizations.append(
                        {
                            'id':          idx,
                            'name':        layer_desc,
                            'vis_type':    'Weights',
                            'vis':  vis,
                            'param_count': parameter_count,
                            'data_stats': {
                                'shape':      data.shape,
                                'mean':       mean,
                                'stddev':     std,
                                'histogram':  hist,
                            }
                        }
                    )
            # sort by layer ID
            visualizations = sorted(visualizations, key=lambda x: x['id'])
        return (predictions, visualizations)

    def get_layer_statistics(self, data):
        """
        Returns statistics for the given layer data:
            (mean, standard deviation, histogram)
                histogram -- [y, x, ticks]

        Arguments:
        data -- a np.ndarray
        """
        # XXX These calculations can be super slow
        mean = np.mean(data)
        std = np.std(data)
        y, x = np.histogram(data, bins=20)
        y = list(y)
        ticks = x[[0, len(x) / 2, -1]]
        x = [(x[i] + x[i + 1]) / 2.0 for i in xrange(len(x) - 1)]
        ticks = list(ticks)
        return (mean, std, [y, x, ticks])

    def after_test_run(self, temp_image_path):
        try:
            os.remove(temp_image_path)
        except OSError:
            pass

    def process_test_output(self, line, predictions, test_category):
        # parse mxnet output
        timestamp, level, message = self.preprocess_output_mxnet(line)

        # return false when unrecognized output is encountered
        if not (level or message):
            return False

        if not message:
            return True

        float_exp = '([-]?inf|nan|[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)'

        # format of output while testing single image
        match = re.match(r'For image \d+, predicted class \d+: \d+ \((.*?)\) %s' % (float_exp), message)
        if match:
            label = match.group(1)
            confidence = match.group(2)
            assert not('inf' in confidence or 'nan' in confidence), \
                'Network reported %s for confidence value. Please check image and network' % label
            confidence = float(confidence)
            predictions.append((label, confidence))
            return True

        # format of output while testing multiple images
        match = re.match(r'Predictions for image \d+: (.*)', message)
        if match:
            values = match.group(1).strip()
            # 'values' should contain a JSON representation of
            # the prediction
            predictions.append(eval(values))
            return True

        # path to visualization file
        match = re.match(r'Saving visualization to (.*)', message)
        if match:
            self.visualization_file = match.group(1).strip()
            return True

        # displaying info and warn messages as we aren't maintaining separate log file for model testing
        if level == 'info':
            self.logger.debug('%s classify %s task : %s' % (self.get_framework_id(), test_category, message))
            return True
        if level == 'warning':
            self.logger.warning('%s classify %s task : %s' % (self.get_framework_id(), test_category, message))
            return True

        if level in ['error', 'critical']:
            raise digits.inference.errors.InferenceError(
                '%s classify %s task failed with error message - %s'
                % (self.get_framework_id(), test_category, message))

        return True           # control never reach this line. It can be removed.

    @override
    def infer_many(self, data, snapshot_epoch=None, gpu=None, resize=True):
        # resize parameter is unused
        return self.infer_many_images(data, snapshot_epoch=snapshot_epoch, gpu=gpu)

    def infer_many_images(self, images, snapshot_epoch=None, gpu=None):
        pass

    def has_model(self):
        """
        Returns True if there is a model that can be used
        """
        return len(self.snapshots) != 0

    @override
    def get_model_files(self):
        """
        return paths to model files
        """
        return {
            "Network": self.model_file
        }

    @override
    def get_network_desc(self):
        """
        return text description of network
        """
        with open(os.path.join(self.job_dir, MXNET_MODEL_FILE), "r") as infile:
            desc = infile.read()
        return desc

    @override
    def get_task_stats(self, epoch=-1):
        """
        return a dictionary of task statistics
        """

        loc, mean_file = os.path.split(self.dataset.get_mean_file())

        stats = {
            "image dimensions": self.dataset.get_feature_dims(),
            "mean file": mean_file,
            "snapshot file": self.get_snapshot_filename(epoch),
            "model file": self.model_file,
            "framework": "mxnet"
        }

        if hasattr(self, "digits_version"):
            stats.update({"digits version": self.digits_version})

        if hasattr(self.dataset, "resize_mode"):
            stats.update({"image resize mode": self.dataset.resize_mode})

        if hasattr(self.dataset, "labels_file"):
            stats.update({"labels file": self.dataset.labels_file})

        return stats
