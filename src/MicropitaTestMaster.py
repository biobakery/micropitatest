#######################################################
# Author: Timothy Tickle
# Description: Class to manage all tests in the regression suite
#######################################################

__author__ = "Timothy Tickle"
__copyright__ = "Copyright 2011"
__credits__ = ["Timothy Tickle"]
__license__ = ""
__version__ = ""
__maintainer__ = "Timothy Tickle"
__email__ = "ttickle@sph.harvard.edu"
__status__ = "Development"

#Import libraries
import unittest

#Import test libraries
import MicroPITATest


suite = unittest.TestSuite()
suite.addTest(MicroPITATest.suite())

runner = unittest.TextTestRunner()
runner.run(suite)
