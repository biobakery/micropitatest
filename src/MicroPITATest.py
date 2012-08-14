"""
Author: Timothy Tickle
Description: Class to test the RunMicroPITA class
"""

__author__ = "Timothy Tickle"
__copyright__ = "Copyright 2011"
__credits__ = ["Timothy Tickle"]
__license__ = ""
__version__ = ""
__maintainer__ = "Timothy Tickle"
__email__ = "ttickle@sph.harvard.edu"
__status__ = "Development"

#Import libraries
from ConstantsMicropitaTest import ConstantsMicropitaTest
from micropita.MicroPITA import MicroPITA
from micropita.src.breadcrumbs.AbundanceTable import AbundanceTable
from micropita.src.breadcrumbs.CommandLine import CommandLine
from micropita.src.ConstantsMicropita import ConstantsMicropita
from micropita.src.breadcrumbs.Metric import Metric
from micropita.src.breadcrumbs.MLPYDistanceAdaptor import MLPYDistanceAdaptor
from micropita.src.breadcrumbs.SVM import SVM
from micropita.src.breadcrumbs.UtilityMath import UtilityMath
import mlpy
import numpy as np
import operator
import os
import re
import unittest

class MicroPITATest(unittest.TestCase):

##### Used to help classify accuracy of selection in methods which have stochasticity
    setComplex = (["Sample_0_D","Sample_1_D","Sample_2_D","Sample_3_D","Sample_4_D","Sample_5_D","Sample_6_D","Sample_7_D","Sample_8_D","Sample_9_D","Sample_10_D","Sample_11_D","Sample_12_D","Sample_13_D","Sample_14_D","Sample_15_D"])
    setMaxVariable = (["Sample_30_E","Sample_31_E","Sample_32_E","Sample_33_E","Sample_34_E","Sample_35_E","Sample_36_E","Sample_37_E","Sample_38_E","Sample_39_E","Sample_40_E	Sample_41_E","Sample_42_E","Sample_43_E"])
    setMinimalVariance = (["Sample_16_R","Sample_17_R","Sample_18_R","Sample_19_R","Sample_20_R","Sample_21_R","Sample_22_R","Sample_23_R","Sample_24_R","Sample_25_R","Sample_26_R","Sample_27_R","Sample_28_R","Sample_29_R"])
    setTargetedTaxa = (["Sample_44_T","Sample_45_T","Sample_46_T","Sample_47_T","Sample_16_R","Sample_17_R","Sample_32_E"])

    dictAnswerClasses = {ConstantsMicropita.c_strDiversity1:setComplex,
                         ConstantsMicropita.c_strExtremeDissimiarity1:setMaxVariable+setMinimalVariance+setTargetedTaxa,
                         ConstantsMicropita.c_strUserRanked:setTargetedTaxa,
                         ConstantsMicropita.c_strSVMClose:setComplex,
                         ConstantsMicropita.c_strSVMFar:setMaxVariable+setMinimalVariance+setTargetedTaxa}

#####Test funcGetTopRankedSamples
    def testfuncGetTopRankedSamplesForGoodCase1(self):
        
        #Inputs
        scores = [[1,2,3,4,5,6,7,8,9,10]]
        N = 3

        #Correct Answer
        answer = [[9,8,7]]

        #Call method
        result = MicroPITA().funcGetTopRankedSamples(lldMatrix = scores, iTopAmount = N)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCase2(self):
        
        #Inputs
        scores = [[10,9,8,7,6,5,4,3,2,1]]
        N = 3

        #Correct Answer
        answer = [[0,1,2]]

        #Call method
        result = MicroPITA().funcGetTopRankedSamples(lldMatrix = scores, iTopAmount = N)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCase3(self):
        
        #Inputs
        scores = [[10,3,40,56,35,678,3,0,-2366]]
        N = 3

        #Correct Answer
        answer = [[5,3,2]]

        #Call method
        result = MicroPITA().funcGetTopRankedSamples(lldMatrix = scores, iTopAmount = N)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCase4(self):
        
        #Inputs
        scores = [[.1,.4,.2,.5,.6,.7,.46,.9]]
        N = 3

        #Correct Answer
        answer = [[7,5,4]]

        #Call method
        result = MicroPITA().funcGetTopRankedSamples(lldMatrix = scores, iTopAmount = N)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCase5(self):
        
        #Inputs
        scores = [[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]]
        N = 3

        #Correct Answer
        answer = [[9,8,7]]

        #Call method
        result = MicroPITA().funcGetTopRankedSamples(lldMatrix = scores, iTopAmount = N)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCase6(self):
        
        #Inputs
        scores = [[1,2,3,4,5,6,7,8,9,10],[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1],[.1,.4,.2,.5,.6,.7,.46,.9]]
        N = 3

        #Correct Answer
        answer = [[9,8,7],[9,8,7],[7,5,4]]

        #Call method
        result = MicroPITA().funcGetTopRankedSamples(lldMatrix = scores, iTopAmount = N)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCase6WithSamples(self):
        
        #Inputs
        scores = [[1,2,3,4,5,6,7,8,9,10],[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1],[.1,.4,.2,.5,.6,.7,.46,.9]]
        names = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
        N = 3

        #Correct Answer
        answer = [["Nine","Eight","Seven"],["Nine","Eight","Seven"],["Seven","Five","Four"]]

        #Call method
        result = MicroPITA().funcGetTopRankedSamples(lldMatrix=scores, lsSampleNames=names, iTopAmount=N)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData1InvSimpson(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [microPITA.c_strInverseSimpsonDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 1

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        abndData.funcNormalize()
        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "['Sample_1']"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData1InvSimpson(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strInverseSimpsonDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 1

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        abndData.funcNormalize()
        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "['Sample_1']"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData1InvSimpson(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strInverseSimpsonDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 1

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        abndData.funcNormalize()
        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "['Sample_1']"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData1InvSimpson(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strInverseSimpsonDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 1

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        abndData.funcNormalize()
        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_1']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData2InvSimpson(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strInverseSimpsonDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 2

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        abndData.funcNormalize()
        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_1', 'Sample_2']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData3InvSimpson(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strInverseSimpsonDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 3

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        abndData.funcNormalize()
        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_1', 'Sample_2', 'Sample_3']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData4InvSimpson(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strInverseSimpsonDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 4

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        abndData.funcNormalize()
        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData5InvSimpson(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strInverseSimpsonDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 5

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        abndData.funcNormalize()
        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4', 'Sample_5']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData1Choa(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strChao1Diversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 1

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_50']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData2Choa(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strChao1Diversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 2

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_50', 'Sample_49']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData3Choa(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strChao1Diversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 3

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_50', 'Sample_49', 'Sample_48']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData4Choa(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strChao1Diversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 4

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_50', 'Sample_49', 'Sample_48', 'Sample_47']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData5Choa(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strChao1Diversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 5

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_50', 'Sample_49', 'Sample_48', 'Sample_47', 'Sample_46']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetTopRankedSamplesForGoodCaseAbridgedData5ChoaInvSimpson(self):

        #Inputs
        normalize = True
        microPITA = MicroPITA()
        metric = [ConstantsMicropita.c_strChao1Diversity,ConstantsMicropita.c_strInverseSimpsonDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 5

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        abundance = abndData.funcGetAbundanceCopy()
        sampleNames = abndData.funcGetSampleNames()

        #Get results
        metrics = Metric.funcBuildAlphaMetricsMatrix(npaSampleAbundance = abundance, lsSampleNames = sampleNames, lsDiversityMetricAlpha = metric)
        result = microPITA.funcGetTopRankedSamples(lldMatrix = metrics, lsSampleNames = sampleNames, iTopAmount = iSelectCount)

        #Correct Answer
        answer = "[['Sample_50', 'Sample_49', 'Sample_48', 'Sample_47', 'Sample_46'], ['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4', 'Sample_5']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))


#####Test funcGetBetaMetric Need to test other beta metrics as they come on line
    def testfuncGetBetaMetricForGoodCaseBrayCurtisMetric(self):

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"hq.otu_04-nul-nul-mtd-trn-flt-abridged.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True

        rawData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        microPITA = MicroPITA()
        metric = ConstantsMicropita.c_strBrayCurtisDissimilarity

        #Generate data
        abundance = rawData.funcGetAbundanceCopy()
        sampleNames = rawData.funcGetSampleNames()
        abundance = UtilityMath.funcTransposeDataMatrix(abundance)
        abundance = abundance[1:4]

        #Get results
        #tempAbundancies Abundancies to be measured. Matrix should be where row = samples and columns = organisms
        result = microPITA.funcGetBetaMetric(npadAbundancies=abundance, sMetric=metric)

        #Correct Answer
        answer = "[ 1.  1.  1.]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetBetaMetricForGoodCaseBrayCurtisMetric4(self):

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"hq.otu_04-nul-nul-mtd-trn-flt-abridged.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True

        rawData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        microPITA = MicroPITA()
        metric = ConstantsMicropita.c_strBrayCurtisDissimilarity

        #Generate data
        abundance = rawData.funcGetAbundanceCopy()
        sampleNames = rawData.funcGetSampleNames()
        abundance = UtilityMath.funcTransposeDataMatrix(abundance)
        abundance = abundance[1:5]

        #Get results
        #tempAbundancies Abundancies to be measured. Matrix should be where row = samples and columns = organisms
        result = microPITA.funcGetBetaMetric(npadAbundancies=abundance, sMetric=metric)

        #Correct Answer
        answer = "[ 1.          1.          0.90697674  1.          0.49714286  1.        ]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetBetaMetricForGoodCaseInvBrayCurtisMetric(self):

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"hq.otu_04-nul-nul-mtd-trn-flt-abridged.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        microPITA = MicroPITA()
        metric = ConstantsMicropita.c_strInvBrayCurtisDissimilarity

        rawData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        #Generate data
        abundance = rawData.funcGetAbundanceCopy()
        sampleNames = rawData.funcGetSampleNames()
        abundance = UtilityMath.funcTransposeDataMatrix(abundance)
        abundance = abundance[1:4]

        #Get results
        #tempAbundancies Abundancies to be measured. Matrix should be where row = samples and columns = organisms
        result = microPITA.funcGetBetaMetric(npadAbundancies=abundance, sMetric=metric)

        #Correct Answer
        answer = "[ 0.  0.  0.]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testfuncGetBetaMetricForGoodCaseInvBrayCurtisMetric4(self):

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"hq.otu_04-nul-nul-mtd-trn-flt-abridged.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        microPITA = MicroPITA()
        metric = ConstantsMicropita.c_strInvBrayCurtisDissimilarity

        rawData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        #Generate data
        abundance = rawData.funcGetAbundanceCopy()
        sampleNames = rawData.funcGetSampleNames()
        abundance = UtilityMath.funcTransposeDataMatrix(abundance)
        abundance = abundance[1:5]

        #Get results
        #tempAbundancies Abundancies to be measured. Matrix should be where row = samples and columns = organisms
        result = microPITA.funcGetBetaMetric(npadAbundancies=abundance, sMetric=metric)

        #Correct Answer
        answer = "[ 0.          0.          0.09302326  0.          0.50285714  0.        ]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

#####Test funcGetCentralSamplesByKMedoids
    def testfuncGetCentralSamplesByKMedoidsForGoodCaseBC(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        data = np.array([[0.2,0.8],[1.0,0.0],[0.1,0.9],[.95,0.05],[0.5,0.5],[0.45,0.55],[0.25,0.75],[0.40,0.60],[0.83,0.17]])
        sampleNames = ("One","Two","Three","Four","Five","Six","Seven","Eight","Nine")
        # [0.1,0.9], 3
        #[[0.2,0.8], 1
        # [0.25,0.75], 7
        #
        # [0.5,0.5], 5
        # [0.45,0.55], 6
        # [0.40,0.60], 8
        #
        # [0.83,0.17]] 9
        # [.95,0.05], 4
        # [1.0,0.0], 2

        numberClusters = 3
        numberSamplesReturned = 3

        #Correct Answer
        answer = "['Six', 'One', 'Four']"

        #Call method
        result = microPITA.funcGetCentralSamplesByKMedoids(npaMatrix=data, sMetric=Metric.c_strBrayCurtisDissimilarity, lsSampleNames = sampleNames, iNumberSamplesReturned = numberSamplesReturned)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

###Test selectExtremeSamplesFromHClust
    def testFuncSelectExtremeSamplesForGoodCase1Different(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strBetaMetric = ConstantsMicropita.c_strInvBrayCurtisDissimilarity
        lsSampleNames = ["One","Two","Three","Four","Five","Six"]
        iSelectSampleCount = 1
        npaAbundanceMatrix = np.array([[1,0,0,0,0],[0,1,1,1,1],[0,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]])

        #Get results
        result = microPITA.funcSelectExtremeSamplesFromHClust(strBetaMetric=strBetaMetric, npaAbundanceMatrix=npaAbundanceMatrix,
                                                 lsSampleNames=lsSampleNames, iSelectSampleCount=iSelectSampleCount)
        result.sort()

        answer = ["One"]
        answer.sort()

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectExtremeSamplesForGoodCase2Different(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strBetaMetric = ConstantsMicropita.c_strInvBrayCurtisDissimilarity
        lsSampleNames = ["One","Two","Three","Four","Five","Six"]
        iSelectSampleCount = 2
        npaAbundanceMatrix = np.array([[1,0,0,0,0],[0,1,1,1,1],[0,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]])

        #Get results
        result = microPITA.funcSelectExtremeSamplesFromHClust(strBetaMetric=strBetaMetric, npaAbundanceMatrix=npaAbundanceMatrix,
                                                 lsSampleNames=lsSampleNames, iSelectSampleCount=iSelectSampleCount)
        result.sort()

        answer = ["One","Three"]
        answer.sort()

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectExtremeSamplesForGoodCase1Different1Groups(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strBetaMetric = ConstantsMicropita.c_strInvBrayCurtisDissimilarity
        lsSampleNames = ["One","Two","Three","Four"]
        iSelectSampleCount = 1
        npaAbundanceMatrix = np.array([[1,0,0,0],
                                       [1,1,1,0],
                                       [1,1,0,0],
                                       [0,1,1,0]])

        #Get results
        result = microPITA.funcSelectExtremeSamplesFromHClust(strBetaMetric=strBetaMetric, npaAbundanceMatrix=npaAbundanceMatrix,
                                                 lsSampleNames=lsSampleNames, iSelectSampleCount=iSelectSampleCount)

        result.sort()

        answer = ["One"]
        answer.sort()

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectExtremeSamplesForGoodCase2GradientGroups2Samples(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strBetaMetric = ConstantsMicropita.c_strInvBrayCurtisDissimilarity
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
        iSelectSampleCount = 2
        npaAbundanceMatrix = np.array([[1,0,0,0,0],
                                       [1,1,0,0,0],
                                       [1,1,1,0,0],
                                       [1,1,1,1,0],
                                       [1,1,1,1,1],
                                       [0,1,1,1,1],
                                       [0,0,1,1,1],
                                       [0,0,0,1,1],
                                       [0,0,0,0,1]])

        #Get results
        result = microPITA.funcSelectExtremeSamplesFromHClust(strBetaMetric=strBetaMetric, npaAbundanceMatrix=npaAbundanceMatrix,
                                                 lsSampleNames=lsSampleNames, iSelectSampleCount=iSelectSampleCount)

        result.sort()

        answer = ["Four","Nine"]
        answer.sort()

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectExtremeSamplesForGoodCase2GradientGroups4Samples(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strBetaMetric = ConstantsMicropita.c_strInvBrayCurtisDissimilarity
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
        iSelectSampleCount = 4
        npaAbundanceMatrix = np.array([[1,0,0,0,0],
                                       [1,1,0,0,0],
                                       [1,1,1,0,0],
                                       [1,1,1,1,0],
                                       [1,1,1,1,1],
                                       [0,1,1,1,1],
                                       [0,0,1,1,1],
                                       [0,0,0,1,1],
                                       [0,0,0,0,1]])

        #Get results
        result = microPITA.funcSelectExtremeSamplesFromHClust(strBetaMetric=strBetaMetric, npaAbundanceMatrix=npaAbundanceMatrix,
                                                 lsSampleNames=lsSampleNames, iSelectSampleCount=iSelectSampleCount)

        result.sort()

        answer = ["Three","Four","Eight","Nine"]
        answer.sort()

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectExtremeSamplesForGoodCase2GradientGroups6Samples(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strBetaMetric = ConstantsMicropita.c_strInvBrayCurtisDissimilarity
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]
        iSelectSampleCount = 6
        npaAbundanceMatrix = np.array([[1,0,0,0,0],
                                       [1,1,0,0,0],
                                       [1,1,1,0,0],
                                       [1,1,1,1,0],
                                       [1,1,1,1,1],
                                       [0,1,1,1,1],
                                       [0,0,1,1,1],
                                       [0,0,0,1,1],
                                       [0,0,0,0,1]])

        #Get results
        result = microPITA.funcSelectExtremeSamplesFromHClust(strBetaMetric=strBetaMetric, npaAbundanceMatrix=npaAbundanceMatrix,
                                                 lsSampleNames=lsSampleNames, iSelectSampleCount=iSelectSampleCount)

        result.sort()

        answer = ["Two","Three","Four","Seven","Eight","Nine"]
        answer.sort()

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectExtremeSamplesForGoodCase3Different3Groups(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strBetaMetric = ConstantsMicropita.c_strInvBrayCurtisDissimilarity
        lsSampleNames = ["One","Two","Three","Four","Five","Six"]
        iSelectSampleCount = 3
        npaAbundanceMatrix = np.array([[1,0,0,0],
                                       [0,0,0,1],
                                       [1,0,0,1],
                                       [1,1,0,0],
                                       [0,0,1,1],
                                       [1,1,1,1]])

        #Get results
        result = microPITA.funcSelectExtremeSamplesFromHClust(strBetaMetric=strBetaMetric, npaAbundanceMatrix=npaAbundanceMatrix,
                                                 lsSampleNames=lsSampleNames, iSelectSampleCount=iSelectSampleCount)

        result.sort()

        answer = ["One","Four","Five"]
        answer.sort()

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))


###Test funcGetAverageAbundanceSamples
    def testfuncGetAverageAbundanceSamplesForGoodCase1Feature(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]
        fRanked = False

        answer= [["700098980","12.0",1],["700037470","6.0",1],["700098986","1.0",1],["700098988","1.0",1],["700098982","0.0",1]]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def nntestfuncGetAverageAbundanceSamplesForGoodCase1FeatureRanked(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]
        fRanked = True

        answer= [["700037470","2.0",6.0],["700098986","3.0",1.0],["700098980","4.0",12.0],["700098988","5.0",1.0],["700098982","9.0",0.0]]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncGetAverageAbundanceSamplesForGoodCase2Feature(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        fRanked = False

        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72",
                      "Bacteria|Firmicutes|Bacilli|Lactobacillales|Lactobacillaceae|Lactobacillus|1361"]
        answer= [["700037470","25.5",1],["700098980","20.5",1],["700098986","2.0",1],["700098988","2.0",1],["700098982","0.0",1]]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def nntestfuncGetAverageAbundanceSamplesForGoodCase2FeatureRanked(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        fRanked = True

        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72",
                      "Bacteria|Firmicutes|Bacilli|Lactobacillales|Lactobacillaceae|Lactobacillus|1361"]
        answer= [["700037470","1.5",25.5],["700098986","2.5",2.0],["700098980","3.5",20.5],["700098988","4.0",2.0],["700098982","9.0",0.0]]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncGetAverageAbundanceSamplesForGoodCaseAllFeature(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        fRanked = False
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72",
                      "Bacteria|Firmicutes|Bacilli|Lactobacillales|Lactobacillaceae|Lactobacillus|1361",
                      "Bacteria|unclassified|4904","Bacteria|Firmicutes|Bacilli|Bacillales|Bacillaceae|unclassified|1368",
                      "Bacteria|3417"]
        answer= [["700098980","24.0",1],["700037470","11.4",1],["700098988","3.0",1],["700098986","1.8",1],["700098982","0.0",1]]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def nntestfuncGetAverageAbundanceSamplesForGoodCaseAllFeatureRankedWithTie(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        fRanked = True
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72",
                      "Bacteria|Firmicutes|Bacilli|Lactobacillales|Lactobacillaceae|Lactobacillus|1361",
                      "Bacteria|unclassified|4904","Bacteria|Firmicutes|Bacilli|Bacillales|Bacillaceae|unclassified|1368",
                      "Bacteria|3417"]
        answer= [["700037470","2.2",11.4],["700098986","2.8",1.8],["700098980","3.0",24.0],["700098988","3.0",3.0],["700098982","9.0",0.0]]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def nntestfuncGetAverageAbundanceSamplesForGoodCaseAllFeatureRankedWithTies2(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        fRanked = True
        liFeatures = ["Bacteria|unclassified|4904"]
        answer= [["700098980","1.0",43.0],["700098988","4.0",2.0],["700098986","6.0",0.0],["700037470","7.0",0.0],["700098982","9.0",0.0]]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

###Test funcSelectTargetedTaxaSamples
    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect1(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 1
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect2(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 2
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980","700037470"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect3(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 3
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980","700037470","700098986"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect4(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 4
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980","700037470","700098986","700098988"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect5(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 5
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980","700037470","700098986","700098988","700098982"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect1Ranked(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 1
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect2Ranked(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 2
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470","700098986"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect3Ranked(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 3
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470","700098986","700098980"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect4Ranked(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 4
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470","700098986","700098980","700098988"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncSelectTargetedTaxaSamplesForGoodCase1FeatureSelect5Ranked(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.c_cTab
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 5
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470","700098986","700098980","700098988","700098982"]

        abndData = AbundanceTable.funcMakeFromFile(strInputFile=inputFile, fIsNormalized=fIsNormalized, fIsSummed=fIsSummed,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcSelectTargetedTaxaSamples(abndMatrix=abndData, lsTargetedTaxa=liFeatures, iSampleSelectionCount=iSampleCount, sMethod=sMethod)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

##### Test funcGetRandomSamples
    def testfuncGetRandomSamplesForGoodCase1of10Samples(self):
        
        #Inputs
        samples = ["one","two","three","four","five","six","seven","eight","nine","ten"]
        N = 1

        #Correct Answer
        answer = False

        #Tracking error information
        foundError = False
        errorString = ""

        #Call method
        result = MicroPITA().funcGetRandomSamples(lsSamples = samples, iNumberOfSamplesToReturn = N)

        #Check to make sure only N samples are returned
        samplesReturnLength=len(result)

        #Make sure the samples are different (this assumes the samples were initially unique in the test).
        uniqueCounts = len(set(result))

        #Evaluate
        if(samplesReturnLength != N):
            foundError = True
            errorString = errorString + "Should have been "+str(N)+" but received "+str(samplesReturnLength)+"."
        if(uniqueCounts != N):
            foundError = True
            errorString = errorString + "Should have had a unique count of "+str(N)+" but instead received an element count of "+str(uniqueCounts)+". Elements = "+str(result)

        #Check result against answer
        self.assertEqual(str(foundError),str(answer),"".join([str(self),"::",str(errorString),"."]))

    def testfuncGetRandomSamplesForGoodCase3of10Samples(self):
        
        #Inputs
        samples = ["one","two","three","four","five","six","seven","eight","nine","ten"]
        N = 3

        #Correct Answer
        answer = False

        #Tracking error information
        foundError = False
        errorString = ""

        #Call method
        result = MicroPITA().funcGetRandomSamples(lsSamples = samples, iNumberOfSamplesToReturn = N)

        #Check to make sure only N samples are returned
        samplesReturnLength=len(result)

        #Make sure the samples are different (this assumes the samples were initially unique in the test).
        uniqueCounts = len(set(result))

        #Evaluate
        if(samplesReturnLength != N):
            foundError = True
            errorString = errorString + "Should have been "+str(N)+" but received "+str(samplesReturnLength)+"."
        if(uniqueCounts != N):
            foundError = True
            errorString = errorString + "Should have had a unique count of "+str(N)+" but instead received an element count of "+str(uniqueCounts)+". Elements = "+str(result)

        #Check result against answer
        self.assertEqual(str(foundError),str(answer),"".join([str(self),"::",str(errorString),"."]))

    def testfuncGetRandomSamplesForGoodCase10of10Samples(self):
        
        #Inputs
        samples = ["one","two","three","four","five","six","seven","eight","nine","ten"]
        N = 10

        #Correct Answer
        answer = False

        #Tracking error information
        foundError = False
        errorString = ""

        #Call method
        result = MicroPITA().funcGetRandomSamples(lsSamples = samples, iNumberOfSamplesToReturn = N)

        #Check to make sure only N samples are returned
        samplesReturnLength=len(result)

        #Make sure the samples are different (this assumes the samples were initially unique in the test).
        uniqueCounts = len(set(result))

        #Evaluate
        if(samplesReturnLength != N):
            foundError = True
            errorString = errorString + "Should have been "+str(N)+" but received "+str(samplesReturnLength)+"."
        if(uniqueCounts != N):
            foundError = True
            errorString = errorString + "Should have had a unique count of "+str(N)+" but instead received an element count of "+str(uniqueCounts)+". Elements = "+str(result)

        #Check result against answer
        self.assertEqual(str(foundError),str(answer),"".join([str(self),"::",str(errorString),"."]))

    ##### funcRunSVM
    def nottestfuncRunSVM(self):

        #Inputs
        #Reading file
        inputFile="./input/micropita/src/Testing/Data/AbridgedDocuments/hq.otu_04-nul-nul-mtd-trn-flt-abridged.txt"
        outputFile="./input/micropita/src/Testing/Data/AbridgedDocuments/hq.otu_04-nul-nul-mtd-trn-flt-abridged.SVM.txt"
        delimiter=ConstantsMicropita.c_cTab
        nameRow=0
        sLastMetadataName="STSite"
        skipColumn1=True
        normalize=True
        lowestScaleBound=0
        probabilistic=True
        cRange="-5,-4,-3,-2,-1,0,1,2,3,4,5"
        gRange="-5,-4,-3,-2,-1,0,1,2,3,4,5"

        #Inputs  programmatic
        microPITA = MicroPITA()
        labels = [0,0,0,0,0,1,1,1,1,1]

        #Generate data
        microPITA.funcRunSVM(tempInputFile=inputFile, tempDelimiter=ConstantsMicropita.c_cTab, tempOutputSVMFile=outputFile, tempMatrixLabels=labels, sLastMetadataName=sLastMetadataName, tempSkipFirstColumn=skipColumn1, tempNormalize=normalize, tempSVMScaleLowestBound=lowestScaleBound, tempSVMLogG=gRange, tempSVMLogC=cRange, tempSVMProbabilistic=probabilistic)

        #Get results
        result = ""

        #Correct Answer
        answer = "[['700037472', '700098984'], ['700037476', '700098980'], ['700037476', '700098980']]"

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

### Test runMLPYSVM, should be tested in funcRun tests

    def testFuncStoreSVMProbabilityForGoodCaseNoPriorStateProbabilistic(self):

        #Input
        lsValidationSamples = ["Sample1","Sample2","Sample3","Sample4","Sample5"]
        lSVMLabels = [0,1,2]
        ldValidationLabels = [2,1,0,1,2]
        lPredictions = [0,1,2,1,1]
        npaDistances = [[0.0,.7,.3],[.0,.5,.5],[0.0,0.0,1.0],[.4,.1,.5],[.9,.05,.05]]
        dictdProbability = dict()
        dictAllProbabilities = dict()
        dictiPrediction = dict()
        dictAllPredictions = dict()

        #Make results
        lreturn = MicroPITA()._funcStoreSVMProbability(lsValidationSamples,
                                                       ldValidationLabels,
                                                       lSVMLabels,
                                                       npaDistances,
                                                       lPredictions,
                                                       dictdProbability,
                                                       dictAllProbabilities,
                                                       dictiPrediction,
                                                       dictAllPredictions)
        dictdProbability,dictAllProbabilities,dictiPrediction,dictAllPredictions = lreturn

        result = " ".join([str([key,dictdProbability[key]]) for key in sorted(dictdProbability.keys())])
        result = result+", "+" ".join([str([key,dictAllProbabilities[key]]) for key in sorted(dictAllProbabilities.keys())])
        result = result+", "+" ".join([str([key,dictiPrediction[key]]) for key in sorted(dictiPrediction.keys())])
        result = result+", "+" ".join([str([key,dictAllPredictions[key]]) for key in sorted(dictAllPredictions.keys())])

        #Correct answers
        dictdProbabilityAnswer = {"Sample1":.7,"Sample2":.5,"Sample3":1.0,"Sample4":.5,"Sample5":.9}
        dictAllProbabilitiesAnswer = {"Sample1":[1,0.0,.7,.3],"Sample2":[1,.0,.5,.5],"Sample3":[2,0.0,0.0,1.0],"Sample4":[2,.4,.1,.5],"Sample5":[0,.9,.05,.05]}
        dictiPredictionAnswer = {"Sample1":False,"Sample2":True,"Sample3":False,"Sample4":False,"Sample5":False}
        dictAllPredictionsAnswer = {"Sample1":"1","Sample2":"1","Sample3":"2","Sample4":"2","Sample5":"0"}

        answer = " ".join([str([key,dictdProbabilityAnswer[key]]) for key in sorted(dictdProbabilityAnswer.keys())])
        answer = answer+", "+" ".join([str([key,dictAllProbabilitiesAnswer[key]]) for key in sorted(dictAllProbabilitiesAnswer.keys())])
        answer = answer+", "+" ".join([str([key,dictiPredictionAnswer[key]]) for key in sorted(dictiPredictionAnswer.keys())])
        answer = answer+", "+" ".join([str([key,dictAllPredictionsAnswer[key]]) for key in sorted(dictAllPredictionsAnswer.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncStoreSVMProbabilityForGoodCaseWithPriorStateProbabilistic(self):

        #Input
        lsValidationSamples = ["Sample1","Sample2","Sample3","Sample4","Sample5"]
        lSVMLabels = [0,1,2]
        ldValidationLabels = [2,1,0,1,2]
        lPredictions = [0,1,2,1,1]
        npaDistances = [[0.0,.7,.3],[.0,.5,.5],[0.0,0.0,1.0],[.4,.1,.5],[.9,.05,.05]]
        dictdProbability = {"Sample0":.99,"Sample10":.4}
        dictAllProbabilities = {"Sample0":[.99,.1,.0],"Sample10":[.1,.4,.5]}
        dictiPrediction = {"Sample0":True,"Sample10":False}
        dictAllPredictions = {"Sample0":0,"Sample10":1}

        #Make results
        lreturn = MicroPITA()._funcStoreSVMProbability(lsValidationSamples,
                                                       ldValidationLabels,
                                                       lSVMLabels,
                                                       npaDistances,
                                                       lPredictions,
                                                       dictdProbability,
                                                       dictAllProbabilities,
                                                       dictiPrediction,
                                                       dictAllPredictions)
        dictdProbability,dictAllProbabilities,dictiPrediction,dictAllPredictions = lreturn

        result = " ".join([str([key,dictdProbability[key]]) for key in sorted(dictdProbability.keys())])
        result = result+", "+" ".join([str([key,dictAllProbabilities[key]]) for key in sorted(dictAllProbabilities.keys())])
        result = result+", "+" ".join([str([key,dictiPrediction[key]]) for key in sorted(dictiPrediction.keys())])
        result = result+", "+" ".join([str([key,dictAllPredictions[key]]) for key in sorted(dictAllPredictions.keys())])

        #Correct answers
        dictdProbabilityAnswer = {"Sample0":.99,"Sample10":.4,"Sample1":.7,"Sample2":.5,"Sample3":1.0,"Sample4":.5,"Sample5":.9}
        dictAllProbabilitiesAnswer = {"Sample0":[.99,.1,.0],"Sample10":[.1,.4,.5],"Sample1":[1,0.0,.7,.3],"Sample2":[1,.0,.5,.5],"Sample3":[2,0.0,0.0,1.0],"Sample4":[2,.4,.1,.5],"Sample5":[0,.9,.05,.05]}
        dictiPredictionAnswer = {"Sample0":True,"Sample10":False,"Sample1":False,"Sample2":True,"Sample3":False,"Sample4":False,"Sample5":False}
        dictAllPredictionsAnswer = {"Sample0":0,"Sample10":1,"Sample1":"1","Sample2":"1","Sample3":"2","Sample4":"2","Sample5":"0"}

        answer = " ".join([str([key,dictdProbabilityAnswer[key]]) for key in sorted(dictdProbabilityAnswer.keys())])
        answer = answer+", "+" ".join([str([key,dictAllProbabilitiesAnswer[key]]) for key in sorted(dictAllProbabilitiesAnswer.keys())])
        answer = answer+", "+" ".join([str([key,dictiPredictionAnswer[key]]) for key in sorted(dictiPredictionAnswer.keys())])
        answer = answer+", "+" ".join([str([key,dictAllPredictionsAnswer[key]]) for key in sorted(dictAllPredictionsAnswer.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

### test runSupervisedMethods, should be tested in funcRun tests

### test _funcSelectSupervisedSamplesFromPredictFile
    def testFuncSelectSupervisedSamplesFromPredictFile2Classes(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 1
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["Eight","One"],ConstantsMicropita.c_strSVMFar:["Three","Seven"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.464303 0.535697",
                                                      "1 0.510597 0.489403",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "0 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "0 1:0.464303 2:0.535697",
                                                      "1 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "0 1:0.514484",
                                                      "1 1:0.38026 2:0.61974",
                                                      "0 1:0.388271 2:0.611729",
                                                      "1 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile2ClassesDistinct(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = False
        fSelectDistinct = True
        iSelectCount = 1
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMFar:["Three","Seven"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.464303 0.535697",
                                                      "1 0.510597 0.489403",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "0 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "0 1:0.464303 2:0.535697",
                                                      "1 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "0 1:0.514484",
                                                      "1 1:0.38026 2:0.61974",
                                                      "0 1:0.388271 2:0.611729",
                                                      "1 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile2ClassesDiscriminant(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = False
        iSelectCount = 1
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["Eight","One"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.464303 0.535697",
                                                      "1 0.510597 0.489403",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "0 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "0 1:0.464303 2:0.535697",
                                                      "1 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "0 1:0.514484",
                                                      "1 1:0.38026 2:0.61974",
                                                      "0 1:0.388271 2:0.611729",
                                                      "1 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile2ClassesReturn8(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 2
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["Eight","One","Six","Nine"],ConstantsMicropita.c_strSVMFar:["Four","Five","Seven","Ten"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.464303 0.535697",
                                                      "1 0.510597 0.489403",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "1 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "0 1:0.464303 2:0.535697",
                                                      "1 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "1 1:0.514484",
                                                      "0 1:0.38026 2:0.61974",
                                                      "1 1:0.388271 2:0.611729",
                                                      "1 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile2ClassesReturnAll(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 2
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["One","Four","Six","Seven"],ConstantsMicropita.c_strSVMFar:["Two","Three","Five","Eight"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "1 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "0 1:0.464303 2:0.535697",
                                                      "1 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "1 1:0.388271 2:0.611729",
                                                      "1 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile2ClassesReturnMoreThanExists(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 9
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["One","Four","Six","Seven","Two","Three","Five","Eight"],ConstantsMicropita.c_strSVMFar:["One","Four","Six","Seven","Two","Three","Five","Eight"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "1 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "0 1:0.464303 2:0.535697",
                                                      "1 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "1 1:0.388271 2:0.611729",
                                                      "1 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile2ClassesReturnMoreThanExistsMislabeled(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 9
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["One","Four","Six","Seven","Five","Eight"],ConstantsMicropita.c_strSVMFar:["One","Four","Six","Seven","Five","Eight"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "1 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "1 1:0.464303 2:0.535697",
                                                      "0 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "1 1:0.388271 2:0.611729",
                                                      "1 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile2ClassesReturnMoreThanExistsClassMislabeled(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 9
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["Three","Five","Six","Seven"],ConstantsMicropita.c_strSVMFar:["Three","Five","Six","Seven"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "1 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["1 1:0.482089 2:0.517911",
                                                      "1 1:0.464303 2:0.535697",
                                                      "1 1:0.510597",
                                                      "1 1:0.409344 2:0.590656",
                                                      "1 1:0.388271 2:0.611729",
                                                      "1 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "1 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile2ClassesReturnMoreThanExistsClassMislabeled2(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 9
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["One","Two","Four","Eight"],ConstantsMicropita.c_strSVMFar:["One","Two","Four","Eight"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "1 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "0 1:0.464303 2:0.535697",
                                                      "0 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "0 1:0.388271 2:0.611729",
                                                      "0 1:0.5 2:0.5",
                                                      "0 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile2ClassesReturnLess0(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 2
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["Two","Six","Seven"],ConstantsMicropita.c_strSVMFar:["Two","Three","Five"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0",
                                                      "0 0.482089 0.517911",
                                                      "0 0.409344 0.590656",
                                                      "1 0.99 0.01",
                                                      "0 0.5 0.5",
                                                      "1 0.988271 0.011729",
                                                      "1 0.5 0.5",
                                                      "1 0.510373 0.489627",
                                                      "0 0.410257 0.589743"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["1 1:0.482089 2:0.517911",
                                                      "0 1:0.464303 2:0.535697",
                                                      "1 1:0.510597",
                                                      "1 1:0.409344 2:0.590656",
                                                      "1 1:0.388271 2:0.611729",
                                                      "1 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "1 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile3Classes(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 1
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["Four","Seven","Six"],ConstantsMicropita.c_strSVMFar:["One","Five","Three"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0 2",
                                                      "0 0.1 0.5",
                                                      "1 0.2 0.8",
                                                      "2 0.99 0.01",
                                                      "0 0.3 0.3",
                                                      "1 0.988271 0.011729",
                                                      "2 0.3 0.3",
                                                      "1 0.5 0.4",
                                                      "0 0.4 0.5"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "1 1:0.464303 2:0.535697",
                                                      "2 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "1 1:0.388271 2:0.611729",
                                                      "2 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile3ClassesSelect2(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 2
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["Four","Eight","Seven","Two","Six","Three"],ConstantsMicropita.c_strSVMFar:["One","Eight","Five","Two","Three","Six"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0 2",
                                                      "0 0.1 0.5",
                                                      "1 0.2 0.8",
                                                      "2 0.99 0.01",
                                                      "0 0.3 0.3",
                                                      "1 0.988271 0.011729",
                                                      "2 0.3 0.3",
                                                      "1 0.5 0.4",
                                                      "0 0.4 0.5"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "1 1:0.464303 2:0.535697",
                                                      "2 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "1 1:0.388271 2:0.611729",
                                                      "2 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncSelectSupervisedSamplesFromPredictFile3ClassesSelect3(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        strPredictFilePath = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromPredictionFile.txt"])
        strOriginalInputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TestFuncSelectSupervisedSamplesFromInputFile.txt"])
        fSelectDiscriminant = True
        fSelectDistinct = True
        iSelectCount = 3
        lsSampleNames = ["One","Two","Three","Four","Five","Six","Seven","Eight"]

        #Answer
        answer = {ConstantsMicropita.c_strSVMClose:["One","Two","Three","Four","Five","Six","Seven","Eight"],ConstantsMicropita.c_strSVMFar:["One","Two","Three","Four","Five","Six","Seven","Eight"]}

        #Set up. Write rpedict and input files
        strPredFileContents = ConstantsMicropita.c_strEndline.join(["labels 1 0 2",
                                                      "0 0.1 0.5",
                                                      "1 0.2 0.8",
                                                      "2 0.99 0.01",
                                                      "0 0.3 0.3",
                                                      "1 0.988271 0.011729",
                                                      "2 0.3 0.3",
                                                      "1 0.5 0.4",
                                                      "0 0.4 0.5"])
        strInputFileContents = ConstantsMicropita.c_strEndline.join(["0 1:0.482089 2:0.517911",
                                                      "1 1:0.464303 2:0.535697",
                                                      "2 1:0.510597",
                                                      "0 1:0.409344 2:0.590656",
                                                      "1 1:0.388271 2:0.611729",
                                                      "2 1:0.5 2:0.5",
                                                      "1 2:0.489627",
                                                      "0 1:0.410257 2:0.589743"])

        with open(strPredictFilePath, 'w') as f, open(strOriginalInputFile, 'w') as g:
            f.write(strPredFileContents)
            g.write(strInputFileContents)

        #Get answer
        result = microPITA._funcSelectSupervisedSamplesFromPredictFile(strOriginalInputFile=strOriginalInputFile, strPredictFilePath=strPredictFilePath,
                                                                         lsSampleNames=lsSampleNames, iSelectCount=iSelectCount, 
                                                                         fSelectDiscriminant=fSelectDiscriminant, fSelectDistinct=fSelectDistinct)

        #Standardize answer and result
        answer = [[strKey,sorted(answer[strKey])] for strKey in answer]
        result = [[strKey,sorted(result[strKey])] for strKey in result]

        #Delete predict file
        for strFile in [strPredictFilePath,strOriginalInputFile]:
            if os.path.exists(strFile):
                os.remove(strFile)

        #Check result
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

### TestfuncRunNormalizeSensitiveMethods, should be tested in funcRun tests

### Test funcWriteSelectionToFile
    def testFuncWriteSelectionToFileForGoodCase(self):

        #Micropita object
        microPITA = MicroPITA()

        dictTest = {"Diversity_C":["Sample_0_D","Sample_1_D","Sample_2_D","Sample_3_D","Sample_4_D","Sample_5_D"],
		"Distinct":["Sample_41_E","Sample_42_E","Sample_43_E","Sample_45_T","Sample_46_T","Sample_47_T"],
		"Extreme_B":["Sample_7_D","Sample_38_E","Sample_8_D","Sample_43_E","Sample_6_D","Sample_39_E"],
		"Discriminant":["Sample_3_D","Sample_5_D","Sample_6_D","Sample_0_D","Sample_1_D","Sample_2_D"],
		"Representative_B":["Sample_38_E","Sample_39_E","Sample_40_E","Sample_43_E","Sample_44_T","Sample_47_T"],
		"Diversity_I":["Sample_45_T","Sample_44_T","Sample_46_T","Sample_13_D","Sample_9_D","Sample_2_D"],
		"Taxa_Defined":["Sample_47_T","Sample_46_T","Sample_44_T","Sample_45_T","Sample_24_R","Sample_19_R"]}
        lsKeys = ["Diversity_C","Distinct","Extreme_B","Discriminant","Representative_B","Diversity_I","Taxa_Defined"]
        sTestFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TempTestSelectFile.txt"])
        sAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"TestSelectFile.txt"])
        answer = ""

        if os.path.exists(sTestFile):
            os.remove(sTestFile)

        microPITA.funcWriteSelectionToFile(dictSelection=dictTest,strOutputFilePath=sTestFile)

        #Read in generated file and answer
        result = ""
        with open(sTestFile) as f, open(sAnswerFile) as g:
            result = f.read()
            answer = g.read()

        if os.path.exists(sTestFile):
            os.remove(sTestFile)

        #Put answer in correct order
        dictresult = dict([(sLine.split(ConstantsMicropita.c_cTab)[0],sLine.split(ConstantsMicropita.c_cTab)[1:]) for sLine in filter(None,result.split(ConstantsMicropita.c_strEndline))])
        result = ConstantsMicropita.c_strEndline.join([ConstantsMicropita.c_cTab.join([sKey]+dictresult[sKey]) for sKey in lsKeys])+ConstantsMicropita.c_strEndline

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

### Test funcReadSelectionFileToDictionary
    def testFuncReadSelectionFileToDictionaryForGoodCase(self):

        #Micropita object
        microPITA = MicroPITA()

        dictTest = {"Distinct":["Sample_41_E","Sample_42_E","Sample_43_E","Sample_45_T","Sample_46_T","Sample_47_T"],
		"Extreme_B":["Sample_7_D","Sample_38_E","Sample_8_D","Sample_43_E","Sample_6_D","Sample_39_E"],
		"Discriminant":["Sample_3_D","Sample_5_D","Sample_6_D","Sample_0_D","Sample_1_D","Sample_2_D"],
		"Representative_B":["Sample_38_E","Sample_39_E","Sample_40_E","Sample_43_E","Sample_44_T","Sample_47_T"],
		"Diversity_I":["Sample_45_T","Sample_44_T","Sample_46_T","Sample_13_D","Sample_9_D","Sample_2_D"],
		"Taxa_Defined":["Sample_47_T","Sample_46_T","Sample_44_T","Sample_45_T","Sample_24_R","Sample_19_R"]}
        lsKeys = ["Distinct","Extreme_B","Discriminant","Representative_B","Diversity_I","Taxa_Defined"]
        sTestFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"TestSelectFile.txt"])
        answer = "".join(["".join([sKey,str(dictTest[sKey])]) for sKey in lsKeys])

        dictResults = microPITA.funcReadSelectionFileToDictionary(sTestFile)

        #Put answer in correct order
        result = "".join(["".join([sKey,str(dictResults[sKey])]) for sKey in lsKeys])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncReadSelectionFileToDictionaryAndFuncWriteSelectionToFileForGoodCase(self):

        #Micropita object
        microPITA = MicroPITA()

        dictTest = {"Diversity_C":["Sample_0_D","Sample_1_D","Sample_2_D","Sample_3_D","Sample_4_D","Sample_5_D"],
		"Distinct":["Sample_41_E","Sample_42_E","Sample_43_E","Sample_45_T","Sample_46_T","Sample_47_T"],
		"Extreme_B":["Sample_7_D","Sample_38_E","Sample_8_D","Sample_43_E","Sample_6_D","Sample_39_E"],
		"Discriminant":["Sample_3_D","Sample_5_D","Sample_6_D","Sample_0_D","Sample_1_D","Sample_2_D"],
		"Representative_B":["Sample_38_E","Sample_39_E","Sample_40_E","Sample_43_E","Sample_44_T","Sample_47_T"],
		"Diversity_I":["Sample_45_T","Sample_44_T","Sample_46_T","Sample_13_D","Sample_9_D","Sample_2_D"],
		"Taxa_Defined":["Sample_47_T","Sample_46_T","Sample_44_T","Sample_45_T","Sample_24_R","Sample_19_R"]}
        lsKeys = ["Diversity_C","Distinct","Extreme_B","Discriminant","Representative_B","Diversity_I","Taxa_Defined"]
        sTestFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TempTestSelectFile.txt"])
        answer = "".join(["".join([sKey,str(dictTest[sKey])]) for sKey in lsKeys])

        #Get result
        microPITA.funcWriteSelectionToFile(dictSelection=dictTest,strOutputFilePath=sTestFile)
        dictResults = microPITA.funcReadSelectionFileToDictionary(sTestFile)
        #Put answer in correct order
        result = "".join(["".join([sKey,str(dictResults[sKey])]) for sKey in lsKeys])

        if os.path.exists(sTestFile):
            os.remove(sTestFile)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

### Test funcGetAveragePopulation(abndTable, lfCompress)
    def testFuncGetAveragePopulationForGoodCase(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")
        lfCompress = [True]*10

        #Correct Answer
        answer = [22,83,79,83,15]

        #Call method
        result = MicroPITA().funcGetAveragePopulation(abndTable=abndTable, lfCompress=lfCompress)
        #Make consistant the decimal places
        result = [int(dValue*10) for dValue in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncGetAveragePopulationForGoodCaseCollapsing4(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")
        lfCompress = [True,False,True,False,False,True,True,True,True,False]

        #Correct Answer
        answer = [16,40,83,1,21]

        #Call method
        result = MicroPITA().funcGetAveragePopulation(abndTable=abndTable, lfCompress=lfCompress)
        #Make consistant the decimal places
        result = [int(dValue*10) for dValue in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncGetAveragePopulationForGoodCaseCollapsing6(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")
        lfCompress = [False,True,False,True,True,False,False,False,False,True]

        #Correct Answer
        answer = [30,147,72,205,5]

        #Call method
        result = MicroPITA().funcGetAveragePopulation(abndTable=abndTable, lfCompress=lfCompress)
        #Make consistant the decimal places
        result = [int(dValue*10) for dValue in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))


### Test funcGetDistanceFromAverage(abndTable,ldAverage,lsSamples,lfSelected,lfNotSelected)
    def testFuncGetDistanceFromAverageForGoodCaseAverage4distance6(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        lfSelected = [False,True,False,True,True,False,False,False,False,True]

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")
        ldAverage = MicroPITA().funcGetAveragePopulation(abndTable=abndTable, lfCompress=lfSelected)
        lsSamples = abndTable.funcGetSampleNames()

        #Correct Answer
        #Correct answer from R (vegan 2.0-2)
        #> library(vegan)
        #Average
        #> x = c(3.0,14.75,7.25,20.5,0.5)
        #> y1 = c(1,0,3,0,5)
        #> y2 = c(0.0,0.0,0.0,0.0,0.0)
        #> y3 = c(6.0,0.0,45.0,0.0,6.0)
        #> y4 = c(0.0,23.0,0.0,0.0,0)
        #> y5 = c(2.0,0.0,1.0,0.0,1.0)
        #> y6 = c(1.0,1.0,1.0,1.0,1.0)
        #> vegdist(rbind(x,y1),method='bray')
        #          x
        #y 0.8363636
        #> vegdist(rbind(x,y2),method='bray')
        #   x
        #y2 1
        #> vegdist(rbind(x,y3),method='bray')
        #           x
        #y3 0.7912621
        #> vegdist(rbind(x,y4),method='bray')
        #           x
        #y4 0.5724638
        #> vegdist(rbind(x,y5),method='bray')
        #      x
        #y5 0.86
        #> vegdist(rbind(x,y6),method='bray')
        #           x
        #y6 0.8235294

        answer = [83,100,79,57,86,82]

        #Call method
        result = MicroPITA().funcGetDistanceFromAverage(abndTable=abndTable, ldAverage=ldAverage, lsSamples=lsSamples, lfSelected=[not fValue for fValue in lfSelected])
        result = [int(dValue*100) for dValue in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncGetDistanceFromAverageForGoodCaseAverage6distance4(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        lfSelected = [True,False,True,False,False,True,True,True,True,False]

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")
        ldAverage = MicroPITA().funcGetAveragePopulation(abndTable=abndTable, lfCompress=lfSelected)
        lsSamples = abndTable.funcGetSampleNames()

        #Correct Answer
        #Correct answer from R (vegan 2.0-2)
        #> library(vegan)
        #Average
        #> x = c(1.6666666666666667, 4.0, 8.3333333333333339, 0.16666666666666666, 2.1666666666666665)
        #> y1 = c(0.0,10.0,0.0,45.0,0.0)
        #>            x
        #> y1 0.8831776
        #> y2 = c(12.0,43.0,29.0,34.0,2.0)
        #>            x
        #> y2 0.7628362
        #> y3 = c(0.0,6.0,0.0,3.0,0.0)
        #>            x
        #> y3 0.6710526
        #> y4 = c(0.0,0.0,0.0,0.0,0.0)
        #>    x
        #> y4 1
        #> vegdist(rbind(x,y1),method='bray')
        #> vegdist(rbind(x,y2),method='bray')
        #> vegdist(rbind(x,y3),method='bray')
        #> vegdist(rbind(x,y4),method='bray')

        answer = [88,76,67,100]

        #Call method
        result = MicroPITA().funcGetDistanceFromAverage(abndTable=abndTable, ldAverage=ldAverage, lsSamples=lsSamples, lfSelected=[not fValue for fValue in lfSelected])
        result = [int(dValue*100) for dValue in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

#funcMeasureDistanceFromLabelToAverageOtherLabel
    def testFuncMeasureDistanceFromLabelToAverageOtherLabelForGoodCase(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        lfSelected = [True,True,True,True,True,False,False,False,False,False]

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")
        ldAverage = MicroPITA().funcGetAveragePopulation(abndTable=abndTable, lfCompress=lfSelected)

        #Get results
        ltpleSelected, ltpleNotSelected  = MicroPITA().funcMeasureDistanceFromLabelToAverageOtherLabel(abndTable=abndTable,
                                                                                     lfGroup=lfSelected,
                                                                                     lfGroupOther= [not fflage for fflage in lfSelected])

        #Answer
        #Evaluate results
        #Correct answer from R (vegan 2.0-2)
        #> library(vegan)
        #Average
        #> x1 = c(1,0,3,0,5)
        #> x2 = c(0,10,0,45,0)
        #> x3 = c(0,0,0,0,0)
        #> x4 = c(12,43,29,34,2)
        #> x5 = c(0,6,0,3,0)
        #> x6 = c(6,0,45,0,6)
        #> x7 = c(0,23,0,0,0)
        #> x8 = c(2,0,1,0,1)
        #> x9 = c(1,1,1,1,1)
        #> x10 = c(0,0,0,0,0)
        #> average1 = c(2.6,11.8,6.4,16.4,1.4)
        #> average2 = c(1.8,4.8,9.4,0.2,1.6)
        #> vegdist(rbind(average2,x1),method='bray')
        #>     average2
        #> x1 0.5820896
        #> vegdist(rbind(average2,x2),method='bray')
        #>     average2
        #> x2 0.8626374
        #> vegdist(rbind(average2,x3),method='bray')
        #>    average2
        #> x3        1
        #> vegdist(rbind(average2,x4),method='bray')
        #>     average2
        #> x4 0.7416546
        #> vegdist(rbind(average2,x5),method='bray')
        #>     average2
        #> x5 0.6268657
        #> vegdist(rbind(average1,x6),method='bray')
        #>     average1
        #> x6 0.7824268
        #> vegdist(rbind(average1,x7),method='bray')
        #>     average1
        #> x7 0.6168831
        #> vegdist(rbind(average1,x8),method='bray')
        #>     average1
        #> x8 0.8122066
        #> vegdist(rbind(average1,x9),method='bray')
        #>     average1
        #> x9 0.7706422
        #> vegdist(rbind(average1,x10),method='bray')
        #>     average1
        #> x10        1
        ltpleSelectedAnswer = [("700098986",0.5820896),("700098984",0.8626374),("700098982",1),("700098980",0.7416546),("700098988",0.6268657)]
        ltpleNotSelectedAnswer = [("700037470",0.7824268),("700037472",0.6168831),("700037474",0.8122066),("700037476",0.7706422),("700037478",1)]

        #Sort all results and answers
        ltpleSelected.sort(key=operator.itemgetter(0))
        ltpleNotSelected.sort(key=operator.itemgetter(0))
        ltpleSelectedAnswer.sort(key=operator.itemgetter(0))
        ltpleNotSelectedAnswer.sort(key=operator.itemgetter(0))

        #Change doubles to ints so that precision issue will not throw off the tests (perserving to the 4th decimal place)
        ltpleSelected = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in ltpleSelected]
        ltpleNotSelected = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in ltpleNotSelected]
        ltpleSelectedAnswer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in ltpleSelectedAnswer]
        ltpleNotSelectedAnswer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in ltpleNotSelectedAnswer]

        fError = False
        strError = ""
        for iindex, tple in enumerate(ltpleSelected):
            if not (str(ltpleSelectedAnswer[iindex]) == str(ltpleSelected[iindex])):
                fError = True
                strError +=  "".join([str(ltpleSelectedAnswer[iindex])," Did not match ",str(ltpleSelected[iindex])])

        for iindex, tple in enumerate(ltpleNotSelected):
            if not (str(ltpleNotSelectedAnswer[iindex]) == str(ltpleNotSelected[iindex])):
                fError = True
                strError +=  "".join([str(ltpleNotSelectedAnswer[iindex])," Did not match ",str(ltpleNotSelected[iindex])])

        self.assertEqual(False, fError, "".join([str(self),"::",strError,"."]))

#funcPerformDistanceSelection
    #See testFuncMeasureDistanceFromLabelToAverageOtherLabelForGoodCase for calculations in R
    def testFuncPerformDistanceSelectionForGoodCase(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        sLabel = "sLabel"
        iSelect = 2

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")

        lsDisc0, lsDtnt0, lsOther0, lsDisc1, lsDtnt1, lsOther1 = MicroPITA().funcPerformDistanceSelection(abndTable=abndTable, iSelectionCount=iSelect, sLabel=sLabel)

        #Answers
        lsDisc0Answer = [("700098986",0.5820896),("700098988",0.6268657)]
        lsDtnt0Answer = [("700098982",1),("700098984",0.8626374)]
        lsOther0Answer = [("700098980",0.7416546)]
        lsDisc1Answer = [("700037476",0.7706422),("700037472",0.6168831)]
        lsDtnt1Answer = [("700037478",1),("700037474",0.8122066)]
        lsOther1Answer = [("700037470",0.7824268)]

        #Sort answers and results
        lsDisc0Answer.sort(key=operator.itemgetter(0))
        lsDtnt0Answer.sort(key=operator.itemgetter(0))
        lsOther0Answer.sort(key=operator.itemgetter(0))
        lsDisc1Answer.sort(key=operator.itemgetter(0))
        lsDtnt1Answer.sort(key=operator.itemgetter(0))
        lsOther1.sort(key=operator.itemgetter(0))
        lsDisc0.sort(key=operator.itemgetter(0))
        lsDtnt0.sort(key=operator.itemgetter(0))
        lsOther0.sort(key=operator.itemgetter(0))
        lsDisc1.sort(key=operator.itemgetter(0))
        lsDtnt1.sort(key=operator.itemgetter(0))
        lsOther1.sort(key=operator.itemgetter(0))

        #Change doubles to ints so that precision issue will not throw off the tests (perserving to the 4th decimal place)
        lsDisc0Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDisc0Answer]
        lsDtnt0Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDtnt0Answer]
        lsOther0Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsOther0Answer]
        lsDisc1Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDisc1Answer]
        lsDtnt1Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDtnt1Answer]
        lsOther1Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsOther1Answer]
        lsDisc0 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDisc0]
        lsDtnt0 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDtnt0]
        lsOther0 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsOther0]
        lsDisc1 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDisc1]
        lsDtnt1 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDtnt1]
        lsOther1 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsOther1]

        #Result answer pairs
        fError = False
        strError = ""
        llResultAnswers = [[lsDisc1,lsDisc0Answer], [lsDtnt1,lsDtnt0Answer],
                           [lsOther1,lsOther0Answer], [lsDisc0,lsDisc1Answer],
                           [lsDtnt0,lsDtnt1Answer], [lsOther0,lsOther1Answer]]

        for llPairs in llResultAnswers:
            if not (len(llPairs[0]) == len(llPairs[1])):
                fError = True
                strError +=  "".join(["Lengths of answer and result lists did not match:",str(llPairs)])
            else:
                for iindex, tple in enumerate(llPairs[0]):
                    if not (str(llPairs[0][iindex]) == str(llPairs[1][iindex])):
                        fError = True
                        strError +=  "".join([", ",str(llPairs[0][iindex])," Did not match ",str(llPairs[1][iindex])])

        self.assertEqual(False, fError, "".join([str(self),"::",strError,"."]))

#funcRunSupervisedDistancesFromCentroids
#Test for return
    #See testFuncMeasureDistanceFromLabelToAverageOtherLabelForGoodCase for calculations in R
    def testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        strSupervisedMetadata = "sLabel"
        fRunDistinct = True
        fRunDiscriminant = True
        iSelect = 2
        strOuputSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-input.txt"])
        strPredictSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-predict.txt"])

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter="\t", sMetadataID=sSampleIDSelection, sLastMetadata=sLastMetadataSelection, cFeatureNameDelimiter="|")

        dictResults = MicroPITA().funcRunSupervisedDistancesFromCentroids(abundanceTable=abndTable, fRunDistinct=fRunDistinct, fRunDiscriminant=fRunDiscriminant,
                                       strOuputSVMFile=strOuputSVMFile, strPredictSVMFile=strPredictSVMFile, strSupervisedMetadata=strSupervisedMetadata,
                                       iSampleSVMSelectionCount=iSelect)

        dictAnswer = {ConstantsMicropita.c_strDiscriminant:["700098986","700098988","700037476","700037472"],
                      ConstantsMicropita.c_strDistinct:["700098982","700098984","700037478","700037474"]}

        #Check results and answer
        fError = False
        strError = ""
        for sKey in dictAnswer:
            if not sKey in dictResults:
                fError = True
                strError += "".join([" Did not find key in results. Key=",sKey])
            else:
                if not (len(set(dictResults[sKey])&set(dictAnswer[sKey])) == (iSelect * 2)):
                    fError = True
                    strError += "".join([" The results for key=",sKey," were not expected.",str(dictResults[sKey])," Expected:",str(dictAnswer[sKey])])

        self.assertEqual(False, fError, "".join([str(self),"::",strError,"."]))

#Test for created input file
    def testFuncRunSupervisedDistancesFromCentroidsForInputFile(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        strSupervisedMetadata = "sLabel"
        fRunDistinct = True
        fRunDiscriminant = True
        iSelect = 2
        strOuputSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-input.txt"])
        strPredictSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-predict.txt"])
        strCorrectAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-input.txt"])

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter="\t", sMetadataID=sSampleIDSelection, sLastMetadata=sLastMetadataSelection, cFeatureNameDelimiter="|")

        MicroPITA().funcRunSupervisedDistancesFromCentroids(abundanceTable=abndTable, fRunDistinct=fRunDistinct, fRunDiscriminant=fRunDiscriminant,
                                       strOuputSVMFile=strOuputSVMFile, strPredictSVMFile=strPredictSVMFile, strSupervisedMetadata=strSupervisedMetadata,
                                       iSampleSVMSelectionCount=iSelect)

        #Get answer and result
        strAnswer = ""
        strResult = ""
        with open( strCorrectAnswerFile, 'r') as a, open( strOuputSVMFile, 'r') as r:
            strAnswer = a.read()
            strResult = r.read()

        self.assertEqual(str(strResult),str(strAnswer),"".join([str(self),"::Expected=",str(strAnswer),". Received=",str(strResult),"."]))

#Test for created predict file
    def testFuncRunSupervisedDistancesFromCentroidsForPredictFile(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        fSelectionIsNormalized = False
        fSelectionIsSummed = False
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        strSupervisedMetadata = "sLabel"
        fRunDistinct = True
        fRunDiscriminant = True
        iSelect = 2
        strOuputSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-input.txt"])
        strPredictSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-predict.txt"])
        strCorrectAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-predict.txt"])

        abndTable = AbundanceTable.funcMakeFromFile(strInputFile=strSelectionFile, fIsNormalized=fSelectionIsNormalized, fIsSummed=fSelectionIsSummed,
                    cDelimiter="\t", sMetadataID=sSampleIDSelection, sLastMetadata=sLastMetadataSelection, cFeatureNameDelimiter="|")

        MicroPITA().funcRunSupervisedDistancesFromCentroids(abundanceTable=abndTable, fRunDistinct=fRunDistinct, fRunDiscriminant=fRunDiscriminant,
                                       strOuputSVMFile=strOuputSVMFile, strPredictSVMFile=strPredictSVMFile, strSupervisedMetadata=strSupervisedMetadata,
                                       iSampleSVMSelectionCount=iSelect)

        #Get answer and result
        strAnswer = ""
        strResult = ""
        with open( strCorrectAnswerFile, 'r') as a, open( strPredictSVMFile, 'r') as r:
            strAnswer = a.read()
            strResult = r.read()

        self.assertEqual(str(strResult),str(strAnswer),"".join([str(self),"::Expected=",str(strAnswer),". Received=",str(strResult),"."]))

### Test run
    ### Run all Methods
    def nntestFuncRunForGoodCase(self):
        sMethodName = "testFuncRunForGoodCase"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

#        strCheckedFile = 
        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                   fCladesAreSummed=fIsSummed,
                                   sMetadataID=sIDName,
                                   sLastMetadataName=sLastMetadataName,
                                   strInputAbundanceFile=strFileAbund,
                                   strInputPredictFile=strInputPredictFile,
                                   strPredictPredictFile=strPredictPredictFile,
                                   strCheckedAbndFile=strCheckedAbndFile,
                                   strOutputFile=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter = cFeatureNameDelimiter,
                                   strUserDefinedTaxaFile=strFileTaxa,
                                   iSampleSelectionCount=iUnsupervisedSelectionCount,
                                   iSupervisedSampleCount=iSupervisedCount, strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify,
                                   strSelectionTechnique=strSelection,
                                   fSumData=fSumData,
                                   sFeatureSelectionMethod=sFeatureSelection)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)

        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def nntestCallFromCommandlineForGoodCase(self):
        """
        Test commandline call for good case.
        """

        sMethodName = "testCallFromCommandlineForGoodCase"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForGoodCase-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sSumData = [ConstantsMicropita.c_strSumDataArgument]
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sUnSelectionCount = [ConstantsMicropita.c_strUnsupervisedCountArgument,"6"]
        sSelectionCount = [ConstantsMicropita.c_strSupervisedLabelCountArgument,"3"]
        sTaxaFile = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                     "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,
                        ConstantsMicropita.c_strRepresentativeDissimilarity,ConstantsMicropita.c_strTaxa,
                        ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sUnSelectionCount+sSelectionCount+sTaxaFile+sSumData+sSupervisedLabel

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+[strFileAbund,strOutputFile]+strSelection

        #Make sure the output files are not there
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Generate output file
        
        errors = not (commandLine.runCommandLine(lsCommandline) and os.path.exists(strOutputFile))

        #Check for correct output files
        answer = False
        result = errors

        #Read in results and the answer file
        if not errors:
          answer = MicroPITA.funcReadSelectionFileToDictionary(strAnswerFile)
          result = MicroPITA.funcReadSelectionFileToDictionary(strOutputFile)

          #Sort answers
          answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
          result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Delete generated files from test
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseStratify(self):
        sMethodName = "testFuncRunForGoodCaseStratify"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = "Label"
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForErrorCaseNoSelection(self):
        """
        Test handling a scenario where no selection is given.
        """

        sMethodName = "testFuncRunForErrorCaseNoSelection"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 0
        iSupervisedCount = 0
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = False

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForErrorNoExistInputFile(self):
        """
        Test handling a scenario where the input file is specified but does not exist.
        """
        
        sMethodName = "testFuncRunForErrorNoExistInputFile"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingTMP,"AbridgedDocuments/Unbalanced48-GenNoise-0-SignalNoise-5IDONOTEXIST.pcl"])
        if os.path.exists(strFileAbund):
          os.remove(strFileAbund)

        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = False

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForErrorIncorrectID(self):
        """
        Test handling a scenario where the given id is incorrect.
        """
        
        sMethodName = "testFuncRunForErrorIncorrectID"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "BADID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = False

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForErrorIncorrectLastMetadataName(self):
        """
        Test handling a scenario where the given lastMetadataId is incorrect.
        """

        sMethodName = "testFuncRunForErrorIncorrectLastMetadataName"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "BADLabel"
        lsSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Check for exception
        self.assertRaises(ValueError,
                          microPITA.funcRun,fIsNormalized,fIsSummed,sIDName,sLastMetadataName,strFileAbund,
                          strInputPredictFile,strPredictPredictFile,strCheckedAbndFile,strOutFile,
                          cFileDelimiter,cFeatureNameDelimiter,strFileTaxa,iUnsupervisedSelectionCount,
                          iSupervisedCount,lsSelection,sLabel,sUnsupervisedStratify,fSumData,sFeatureSelection)

    def testFuncRunForErrorNoSelection(self):
        """
        Test handling a scenario where no selection technique is given.
        """
        
        sMethodName = "testFuncRunForErrorNoSelection"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = []

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+sMethodName+".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)
        answer = False

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForBadTaxaFileName(self):
        """
        Handling the scenario where the targeted feature file is incorrect / non-existent.
        """
        
        sMethodName = "testFuncRunForBadTaxaFileName"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.badtaxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)
        answer = False

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForNoTaxaFileName(self):
        """
        Handling the scenario where the targeted feature file is not given.
        """
        
        sMethodName = "testFuncRunForNoTaxaFileName"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = None
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)
        answer = False

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForBadLabelName(self):
        """
        Handling the scenario where the label is incorrect.
        """
        
        sMethodName = "testFuncRunForBadLabelName"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "BadLabel"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Answer
        answer = False

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForNoneLabelName(self):
        """
        Handling the scenario where the label is None.
        """
        
        sMethodName = "testFuncRunForNoneLabelName"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = None
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)
        #Answer file
        answer = False

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForNoFeatureSelection(self):
        """
        Handling the scenario where there is no feature selection method indicated.
        """
        
        sMethodName = "testFuncRunForNoFeatureSelection"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,ConstantsMicropita.c_strRepresentativeDissimilarity,
                        ConstantsMicropita.c_strTaxa,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData)

        #Answer 
        answer = False

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForGoodCaseDiversity(self):
        """
        Test running just diversity
        """
        
        sMethodName = "testFuncRunForGoodCaseDiversity"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiversity]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testCallFromCommandlineForGoodCaseDiversity(self):
        """
        Test commandline call for good case only diversity.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseDiversity"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForGoodCaseDiversity-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sSumData = [ConstantsMicropita.c_strSumDataArgument]
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sUnSelectionCount = [ConstantsMicropita.c_strUnsupervisedCountArgument,"6"]
        strSelection = [ConstantsMicropita.c_strDiversity]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sUnSelectionCount+sSumData

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+[strFileAbund,strOutputFile]+strSelection

        #Make sure the output files are not there
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Generate output file
        
        errors = not (commandLine.runCommandLine(lsCommandline) and os.path.exists(strOutputFile))

        #Check for correct output files
        answer = False
        result = errors

        #Read in results and the answer file
        if not errors:
          answer = MicroPITA.funcReadSelectionFileToDictionary(strAnswerFile)
          result = MicroPITA.funcReadSelectionFileToDictionary(strOutputFile)

          #Sort answers
          answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
          result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Delete generated files from test
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))


    def ntestFuncRunForGoodCaseDiversityStratified(self):
        """
        Test running just diversity stratified
        """
        
        sMethodName = "testFuncRunForGoodCaseDiversityStratified"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiversity]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForGoodCaseRepresentative(self):
        """
        Test running just representative
        """
        
        sMethodName = "testFuncRunForGoodCaseRepresentative"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strRepresentativeDissimilarity]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testCallFromCommandlineForGoodCaseRepresentative(self):
        """
        Test commandline call for good case only representative.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseRepresentative"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForGoodCaseRepresentative-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sSumData = [ConstantsMicropita.c_strSumDataArgument]
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sUnSelectionCount = [ConstantsMicropita.c_strUnsupervisedCountArgument,"6"]
        strSelection = [ConstantsMicropita.c_strRepresentativeDissimilarity]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sUnSelectionCount+sSumData

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+[strFileAbund,strOutputFile]+strSelection

        #Make sure the output files are not there
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Generate output file
        
        errors = not (commandLine.runCommandLine(lsCommandline) and os.path.exists(strOutputFile))

        #Check for correct output files
        answer = False
        result = errors

        #Read in results and the answer file
        if not errors:
          answer = MicroPITA.funcReadSelectionFileToDictionary(strAnswerFile)
          result = MicroPITA.funcReadSelectionFileToDictionary(strOutputFile)

          #Sort answers
          answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
          result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Delete generated files from test
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseRepresentativeStratified(self):
        """
        Test running just representative stratified
        """
        
        sMethodName = "testFuncRunForGoodCaseRepresentativeStratified"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strRepresentativeDissimilarity]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForGoodCaseExtreme(self):
        """
        Test running just extreme
        """
        
        sMethodName = "testFuncRunForGoodCaseExtreme"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strExtremeDissimilarity]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testCallFromCommandlineForGoodCaseExtreme(self):
        """
        Test commandline call for good case only Extreme.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseExtreme"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForGoodCaseExtreme-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sSumData = [ConstantsMicropita.c_strSumDataArgument]
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sUnSelectionCount = [ConstantsMicropita.c_strUnsupervisedCountArgument,"6"]
        strSelection = [ConstantsMicropita.c_strExtremeDissimilarity]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sUnSelectionCount+sSumData

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+[strFileAbund,strOutputFile]+strSelection

        #Make sure the output files are not there
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Generate output file
        
        errors = not (commandLine.runCommandLine(lsCommandline) and os.path.exists(strOutputFile))

        #Check for correct output files
        answer = False
        result = errors

        #Read in results and the answer file
        if not errors:
          answer = MicroPITA.funcReadSelectionFileToDictionary(strAnswerFile)
          result = MicroPITA.funcReadSelectionFileToDictionary(strOutputFile)

          #Sort answers
          answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
          result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Delete generated files from test
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseExtremeStratified(self):
        """
        Test running just extreme stratified
        """
        
        sMethodName = "testFuncRunForGoodCaseExtremeStratified"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strExtremeDissimilarity]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForGoodCaseTargetedWithSelectionMethod(self):
        """
        Test running just targeted
        """
        sMethodName = "testFuncRunForGoodCaseTargetedWithSelectionMethod"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strTaxa]

        #Other inputs
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)
        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)

        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testCallFromCommandlineForGoodCaseTargeted(self):
        """
        Test commandline call for good case only targeted.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseTargeted"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForGoodCaseTargetedWithSelectionMethod-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sSumData = [ConstantsMicropita.c_strSumDataArgument]
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sUnSelectionCount = [ConstantsMicropita.c_strUnsupervisedCountArgument,"6"]
        strFileTaxa = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                       "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strTaxa]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sUnSelectionCount+sSumData+strFileTaxa

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+[strFileAbund,strOutputFile]+strSelection

        #Make sure the output files are not there
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Generate output file
        
        errors = not (commandLine.runCommandLine(lsCommandline) and os.path.exists(strOutputFile))

        #Check for correct output files
        answer = False
        result = errors

        #Read in results and the answer file
        if not errors:
          answer = MicroPITA.funcReadSelectionFileToDictionary(strAnswerFile)
          result = MicroPITA.funcReadSelectionFileToDictionary(strOutputFile)

          #Sort answers
          answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
          result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Delete generated files from test
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))


    def ntestFuncRunForGoodCaseTargetedWithSelectionMethodStratified(self):
        """
        Test running just targeted stratified
        """
        
        sMethodName = "testFuncRunForGoodCaseTargetedWithSelectionMethodStratified"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strTaxa]

        #Other inputs
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForGoodCaseTargetedWithoutSelectionMethod(self):
        """
        Test running just targeted
        """

        sMethodName = "testFuncRunForGoodCaseTargetedWithoutSelectionMethod"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = []

        #Other inputs
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)

        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testFuncRunForGoodCaseAllUnsupervised(self):
        sMethodName = "testFuncRunForGoodCaseAllUnsupervised"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,
                        ConstantsMicropita.c_strRepresentativeDissimilarity,ConstantsMicropita.c_strTaxa]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testCallFromCommandlineForGoodCaseUnsupervised(self):
        """
        Test commandline call for good case for unsupervised only.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseUnsupervised"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForGoodCaseAllUnsupervised-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sSumData = [ConstantsMicropita.c_strSumDataArgument]
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sUnSelectionCount = [ConstantsMicropita.c_strUnsupervisedCountArgument,"6"]
        sTaxaFile = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                     "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,
                        ConstantsMicropita.c_strRepresentativeDissimilarity,ConstantsMicropita.c_strTaxa]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sUnSelectionCount+sTaxaFile+sSumData

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+[strFileAbund,strOutputFile]+strSelection

        #Make sure the output files are not there
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Generate output file
        errors = not (commandLine.runCommandLine(lsCommandline) and os.path.exists(strOutputFile))

        #Check for correct output files
        answer = False
        result = errors

        #Read in results and the answer file
        if not errors:
          answer = MicroPITA.funcReadSelectionFileToDictionary(strAnswerFile)
          result = MicroPITA.funcReadSelectionFileToDictionary(strOutputFile)

          #Sort answers
          answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
          result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Delete generated files from test
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))


    def ntestFuncRunForGoodCaseAllUnsupervisedStratfied(self):
        sMethodName = "testFuncRunForGoodCaseAllUnsupervisedStratified"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,
                        ConstantsMicropita.c_strRepresentativeDissimilarity,ConstantsMicropita.c_strTaxa]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        strFileTaxa = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 6
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        strUserDefinedTaxaFile=strFileTaxa,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForSupervisedMethod(self):
        sMethodName = "testFuncRunForSupervisedMethod"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 0
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = False
        error = False
        for sKey in result:
          if len(set(result[sKey])&set(self.dictAnswerClasses[sKey])) == len(result[sKey]):
            error = error and True
          else:
            error = True

        #Check result against answer
        self.assertEqual(str(error),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(error),"."]))

    def testCallFromCommandlineForGoodCaseSupervised(self):
        """
        Test commandline call for good case supervised methods only.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseSupervised"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForSupervisedMethod-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sSumData = [ConstantsMicropita.c_strSumDataArgument]
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strSupervisedLabelCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sSumData+sSupervisedLabel

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+[strFileAbund,strOutputFile]+strSelection

        #Make sure the output files are not there
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Generate output file
        
        errors = not (commandLine.runCommandLine(lsCommandline) and os.path.exists(strOutputFile))

        #Check for correct output files
        answer = False
        result = errors

        #Read in results and the answer file
        if not errors:
          result = MicroPITA.funcReadSelectionFileToDictionary(strOutputFile)

          for sKey in result:
            if len(set(result[sKey])&set(self.dictAnswerClasses[sKey])) == len(result[sKey]):
                errors = errors and True
            else:
                errors = True

        #Delete generated files from test
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Check result against answer
        self.assertEqual(str(errors),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(errors),"."]))


    def testFuncRunForGoodCaseDiscriminantMethod(self):
        sMethodName = "testFuncRunForGoodCaseDiscriminantMethod"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiscriminant]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 0
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,sMethodName,"-Correct.txt"])

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = False
        error = False
        for sKey in result:
          if len(set(result[sKey])&set(self.dictAnswerClasses[sKey])) == len(result[sKey]):
            error = error and True
          else:
            error = True

        #Check result against answer
        self.assertEqual(str(error),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(error),"."]))

    def testCallFromCommandlineForGoodCaseDiscriminant(self):
        """
        Test commandline call for good case discriminant only.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseDiscriminant"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForGoodCaseDiscriminantMethod-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sSumData = [ConstantsMicropita.c_strSumDataArgument]
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strSupervisedLabelCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strDiscriminant]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sSumData+sSupervisedLabel

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+[strFileAbund,strOutputFile]+strSelection

        #Make sure the output files are not there
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Generate output file
        
        errors = not (commandLine.runCommandLine(lsCommandline) and os.path.exists(strOutputFile))

        #Check for correct output files
        answer = False
        result = errors

        #Read in results and the answer file
        if not errors:
          result = MicroPITA.funcReadSelectionFileToDictionary(strOutputFile)

          for sKey in result:
            if len(set(result[sKey])&set(self.dictAnswerClasses[sKey])) == len(result[sKey]):
                errors = errors and True
            else:
                errors = True

        #Delete generated files from test
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Check result against answer
        self.assertEqual(str(errors),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(errors),"."]))

    def testFuncRunForGoodCaseDistinctMethod(self):
        sMethodName = "testFuncRunForGoodCaseDistinctMethod"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        fIsNormalized = False
        fIsSummed = False
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.c_cTab
        cFeatureNameDelimiter = "|"
        strInputPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictInput.txt"])
        strPredictPredictFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictPredict.txt"])
        strCheckedAbndFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,"-predictChecked.txt"])

        iUnsupervisedSelectionCount = 0
        iSupervisedCount = 3
        sLabel = "Label"
        sUnsupervisedStratify = None
        fSumData = False
        sFeatureSelection = ConstantsMicropita.c_strTargetedRanked

        #Get result
        result = microPITA.funcRun(fIsAlreadyNormalized=fIsNormalized,
                                        fCladesAreSummed=fIsSummed,
                                        sMetadataID=sIDName,
                                        sLastMetadataName=sLastMetadataName,
                                        strInputAbundanceFile=strFileAbund,
                                        strInputPredictFile=strInputPredictFile,
                                        strPredictPredictFile=strPredictPredictFile,
                                        strCheckedAbndFile=strCheckedAbndFile,
                                        strOutputFile=strOutFile,
                                        cDelimiter=cFileDelimiter,
                                        cFeatureNameDelimiter = cFeatureNameDelimiter,
                                        iSampleSelectionCount=iUnsupervisedSelectionCount,
                                        iSupervisedSampleCount=iSupervisedCount,
                                        strLabel=sLabel,
                                        strStratify=sUnsupervisedStratify,
                                        strSelectionTechnique=strSelection,
                                        fSumData=fSumData,
                                        sFeatureSelectionMethod=sFeatureSelection)

        answer = False
        error = False
        for sKey in result:
          if len(set(result[sKey])&set(self.dictAnswerClasses[sKey])) == len(result[sKey]):
            error = error and True
          else:
            error = True

        #Check result against answer
        self.assertEqual(str(error),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(error),"."]))

    def testCallFromCommandlineForGoodCaseDistinct(self):
        """
        Test commandline call for good case distinct only.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseDistinct"

        #Commandline object
        commandLine = CommandLine()

        #Answer
        dictAnswer = {}

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sSumData = [ConstantsMicropita.c_strSumDataArgument]
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strSupervisedLabelCountArgument,"3"]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]
        strSelection = [ConstantsMicropita.c_strDistinct]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sSumData+sSupervisedLabel

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+[strFileAbund,strOutputFile]+strSelection

        #Make sure the output files are not there
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Generate output file
        
        errors = not (commandLine.runCommandLine(lsCommandline) and os.path.exists(strOutputFile))

        #Check for correct output files
        answer = False
        result = errors

        #Read in results and the answer file
        if not errors:
          result = MicroPITA.funcReadSelectionFileToDictionary(strOutputFile)

          for sKey in result:
            if len(set(result[sKey])&set(self.dictAnswerClasses[sKey])) == len(result[sKey]):
                errors = errors and True
            else:
                errors = True

        #Delete generated files from test
        for sFile in [strOutputFile]:
          if os.path.exists(sFile):
            os.remove(sFile)

        #Check result against answer
        self.assertEqual(str(errors),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(errors),"."]))
##
#Creates a suite of tests
def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(MicroPITATest)
    return suite
