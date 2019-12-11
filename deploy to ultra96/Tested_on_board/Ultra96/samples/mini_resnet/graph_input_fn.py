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

import numpy as np
import cv2
#Size of images
IMAGE_WIDTH  = 32
IMAGE_HEIGHT = 32

#normalization factor to scale image 0-255 values to 0-1 #DB
NORM_FACTOR = 255.0 

def Normalize(x_test):
    x_test  = np.asarray(x_test)
    x_test = x_test.astype(np.float32)
    x_test = x_test/NORM_FACTOR
    x_test = x_test -0.5
    out_x_test = x_test *2
    return out_x_test


def ScaleTo1(x_test):
    x_test  = np.asarray(x_test)
    x_test = x_test.astype(np.float32)
    our_x_test = x_test/NORM_FACTOR
    return out_x_test

def calib_input(image_path):
  """
  Image pre-process
  """
  """ read image as rgb, returns numpy array (28,28, 3)"""
  image = cv2.imread(image_path)
  
  #image = mean_image_subtraction(image,MEANS)
  image2 = Normalize(image)
  """reshape numpy array"""
  image2 = image2.reshape((image2.shape[0], image2.shape[1], 3))

  return image2
