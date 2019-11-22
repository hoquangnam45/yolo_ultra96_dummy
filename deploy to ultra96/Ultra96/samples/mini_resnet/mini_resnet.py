"""
(c) Copyright 2019 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES. 
"""

import graph_input_fn
from dnndk import n2cube, dputils
import numpy as np
import os

"""DPU Kernel Name for miniResNet"""
KERNEL_CONV="miniResNet_0"

CONV_INPUT_NODE="batch_normalization_1_FusedBatchNorm_1_add"
CONV_OUTPUT_NODE="dense_1_MatMul"

def get_script_directory():
    path = os.getcwd()
    return path

SCRIPT_DIR = get_script_directory()
calib_image_dir  = SCRIPT_DIR + "/../common/image_32_32/"
calib_image_list = calib_image_dir +  "words.txt"

def TopK(dataInput, filePath):
    """
    Get top k results according to its probability
    """
    cnt = [i for i in range(10)]
    pair = zip(dataInput, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    #print(softmax_new,'\n',cnt_new)
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    for i in range(5):
        flag = 0
        for line in data1:
            if flag == cnt_new[i]:
                print("Top[%d] %f %s" %(i, (softmax_new[i]),(line.strip)("\n")))
            flag = flag + 1

def main():

    """ Attach to DPU driver and prepare for running """
    n2cube.dpuOpen()

    """ Create DPU Kernels for CONV NODE in imniResNet """
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)

    """ Create DPU Tasks for CONV NODE in miniResNet """
    task = n2cube.dpuCreateTask(kernel, 0)

    listimage = os.listdir(calib_image_dir)

    for i in range(len(listimage)):
        path = os.path.join(calib_image_dir, listimage[i])
        if os.path.splitext(path)[1] != ".png":
            continue
        print("Loading %s" %listimage[i])

        """ Load image and Set image into CONV Task """
        imageRun=graph_input_fn.calib_input(path)
        imageRun=imageRun.reshape((imageRun.shape[0]*imageRun.shape[1]*imageRun.shape[2]))
        input_len=len(imageRun)
        n2cube.dpuSetInputTensorInHWCFP32(task,CONV_INPUT_NODE,imageRun,input_len)

        """  Launch miniRetNet task """
        n2cube.dpuRunTask(task)

        """ Get output tensor address of CONV """
        conf = n2cube.dpuGetOutputTensorAddress(task, CONV_OUTPUT_NODE)
        
        """ Get output channel of CONV  """
        channel = n2cube.dpuGetOutputTensorChannel(task, CONV_OUTPUT_NODE)
        
        """ Get output size of CONV  """
        size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE)
        
        softmax = [0 for i in range(size)]
       
        """ Get output scale of CONV  """
        scale = n2cube.dpuGetOutputTensorScale(task, CONV_OUTPUT_NODE)
        
        batchSize=size//channel
        """ Calculate softmax and show TOP5 classification result """
        n2cube.dpuRunSoftmax(conf, softmax, channel, batchSize, scale)
        TopK(softmax, calib_image_list)

    """ Destroy DPU Tasks & free resources """
    n2cube.dpuDestroyTask(task)
    """ Destroy DPU Kernels & free resources """
    rtn = n2cube.dpuDestroyKernel(kernel)
    """ Dettach from DPU driver & free resources """
    n2cube.dpuClose()
if __name__ == "__main__":
    main()
