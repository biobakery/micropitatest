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
from micropita.src.breadcrumbs.ConstantsBreadCrumbs import ConstantsBreadCrumbs
from micropita.src.ConstantsMicropita import ConstantsMicropita
from micropita.src.breadcrumbs.Metric import Metric
from micropita.src.breadcrumbs.MLPYDistanceAdaptor import MLPYDistanceAdaptor
from micropita.src.breadcrumbs.SVM import SVM
from micropita.src.breadcrumbs.UtilityMath import UtilityMath
import csv
import mlpy
import numpy as np
import operator
import os
import re
import shutil
import unittest

class MicroPITATest(unittest.TestCase):

##### Used to help classify accuracy of selection in methods which have stochasticity
    setComplex = (["Sample_0_D","Sample_1_D","Sample_2_D","Sample_3_D","Sample_4_D","Sample_5_D","Sample_6_D","Sample_7_D","Sample_8_D","Sample_9_D","Sample_10_D","Sample_11_D","Sample_12_D","Sample_13_D","Sample_14_D","Sample_15_D"])
    setMaxVariable = (["Sample_30_E","Sample_31_E","Sample_32_E","Sample_33_E","Sample_34_E","Sample_35_E","Sample_36_E","Sample_37_E","Sample_38_E","Sample_39_E","Sample_40_E	Sample_41_E","Sample_42_E","Sample_43_E"])
    setMinimalVariance = (["Sample_16_R","Sample_17_R","Sample_18_R","Sample_19_R","Sample_20_R","Sample_21_R","Sample_22_R","Sample_23_R","Sample_24_R","Sample_25_R","Sample_26_R","Sample_27_R","Sample_28_R","Sample_29_R"])
    setTargetedTaxa = (["Sample_44_T","Sample_45_T","Sample_46_T","Sample_47_T","Sample_16_R","Sample_17_R","Sample_32_E"])

    dictAnswerClasses = {ConstantsMicropita.c_strDiversity:setComplex,
                         ConstantsMicropita.c_strExtreme:setMaxVariable+setMinimalVariance+setTargetedTaxa,
                         ConstantsMicropita.c_strFeature:setTargetedTaxa,
                         ConstantsMicropita.c_strDiscriminant:setComplex,
                         ConstantsMicropita.c_strDistinct:setMaxVariable+setMinimalVariance+setTargetedTaxa}

    c_strEndline = os.linesep	

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
        metric = [microPITA.c_strDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 1

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        metric = [ConstantsMicropita.c_strDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 1

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        metric = [ConstantsMicropita.c_strDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 1

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        metric = [ConstantsMicropita.c_strDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 1

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        metric = [ConstantsMicropita.c_strDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 2

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        metric = [ConstantsMicropita.c_strDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 3

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        metric = [ConstantsMicropita.c_strDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 4

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        metric = [ConstantsMicropita.c_strDiversity]

        #Generate data
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"DiversityTest.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Inverse_Simpson"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        iSelectCount = 5

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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

#####Test funcGetBetaMetric Need to test other beta metrics as they come on line
    def testfuncGetBetaMetricForGoodCaseBrayCurtisMetric(self):

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"hq.otu_04-nul-nul-mtd-trn-flt-abridged.txt"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True

        rawData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        microPITA = MicroPITA()
        metric = Metric.c_strBrayCurtisDissimilarity

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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True

        rawData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        microPITA = MicroPITA()
        metric = Metric.c_strBrayCurtisDissimilarity

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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        microPITA = MicroPITA()
        metric = Metric.c_strInvBrayCurtisDissimilarity

        rawData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = True
        microPITA = MicroPITA()
        metric = Metric.c_strInvBrayCurtisDissimilarity

        rawData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        strBetaMetric = Metric.c_strInvBrayCurtisDissimilarity
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
        strBetaMetric = Metric.c_strInvBrayCurtisDissimilarity
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
        strBetaMetric = Metric.c_strInvBrayCurtisDissimilarity
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
        strBetaMetric = Metric.c_strInvBrayCurtisDissimilarity
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
        strBetaMetric = Metric.c_strInvBrayCurtisDissimilarity
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
        strBetaMetric = Metric.c_strInvBrayCurtisDissimilarity
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
        strBetaMetric = Metric.c_strInvBrayCurtisDissimilarity
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]
        fRanked = False

        answer= [["700098980","-1.0",12.0],["700037470","-1.0",6.0],["700098986","-1.0",1.0],["700098988","-1.0",1.0],["700098982","-1.0",0.0]]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncGetAverageAbundanceSamplesForGoodCase1FeatureRanked(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]
        fRanked = True

        answer= [['700037470', '1.5', 6.0], ['700098986', '2.0', 1.0], ['700098982', '2.0', 0.0], ['700098980', '3.0', 12.0], ['700098988', '4.0', 1.0]]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        fRanked = False

        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72",
                      "Bacteria|Firmicutes|Bacilli|Lactobacillales|Lactobacillaceae|Lactobacillus|1361"]
        answer= [["700037470","-1.0",25.5],["700098980","-1.0",20.5],["700098986","-1.0",2.0],["700098988","-1.0",2.0],["700098982","-1.0",0.0]]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncGetAverageAbundanceSamplesForGoodCase2FeatureRanked(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        fRanked = True

        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72",
                      "Bacteria|Firmicutes|Bacilli|Lactobacillales|Lactobacillaceae|Lactobacillus|1361"]
        answer= [['700037470', '0.8', 25.5], ['700098986', '1.5', 2.0], ['700098982', '2.0', 0.0], ['700098980', '2.5', 20.5], ['700098988', '3.0', 2.0]]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
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
        answer= [["700098980","-1.0",24.0],["700037470","-1.0",11.4],["700098988","-1.0",3.0],["700098986","-1.0",1.8],["700098982","-1.0",0.0]]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncGetAverageAbundanceSamplesForGoodCaseAllFeatureRankedWithTie(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        fRanked = True
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72",
                      "Bacteria|Firmicutes|Bacilli|Lactobacillales|Lactobacillaceae|Lactobacillus|1361",
                      "Bacteria|unclassified|4904",
                      "Bacteria|Firmicutes|Bacilli|Bacillales|Bacillaceae|unclassified|1368",
                      "Bacteria|3417"]
        answer= [['700098980', '2.0', 24.0], ['700037470', '2.0', 11.4], ['700098988', '2.0', 3.0], ['700098986', '2.0', 1.8], ['700098982', '2.0', 0.0]]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)

        result = microPITA.funcGetAverageAbundanceSamples(abndTable=abndData, lsTargetedFeature=liFeatures, fRank=fRanked)
        result = [[resultList[0],"{0:.1f}".format(resultList[1]),resultList[2]] for resultList in result]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::\nExpected=\n",str(answer),". \nReceived=\n",str(result),"."]))

    def testfuncGetAverageAbundanceSamplesForGoodCaseAllFeatureRankedWithTies2(self):

        #Micropita object
        microPITA = MicroPITA()

        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"TestFeatureAverages.txt"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        fRanked = True
        liFeatures = ["Bacteria|unclassified|4904"]
        answer= [['700098980', '0.0', 43.0], ['700098982', '2.0', 0.0], ['700098988', '3.0', 2.0], ['700098986', '3.5', 0.0], ['700037470', '3.5', 0.0]]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 1
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 2
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980","700037470"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 3
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980","700037470","700098986"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 4
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980","700037470","700098986","700098988"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 5
        sMethod = ConstantsMicropita.c_strTargetedAbundance
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700098980","700037470","700098986","700098988","700098982"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 1
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 2
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470","700098986"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 3
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470","700098986","700098982"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 4
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470","700098986","700098982","700098980"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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
        delimiter = ConstantsMicropita.TAB
        sNameRow = "TID"
        sLastMetadata = "STSite"
        cFeatureDelimiter = "|"
        fIsSummed = False
        fIsNormalized = False
        iSampleCount = 5
        sMethod = ConstantsMicropita.c_strTargetedRanked
        liFeatures = ["Bacteria|Firmicutes|Clostridia|Clostridiales|Clostridiaceae|Clostridium|72"]

        answer= ["700037470","700098986","700098982","700098980","700098988"]

        abndData = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
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

### test runSupervisedMethods, should be tested in funcRun tests

### TestfuncRunNormalizeSensitiveMethods, should be tested in funcRun tests

### Test funcWriteSelectionToFile
    def testFuncWriteSelectionToFileForGoodCase(self):

        #Micropita object
        microPITA = MicroPITA()

        dictTest = {"distinct":["Sample_41_E","Sample_42_E","Sample_43_E","Sample_45_T","Sample_46_T","Sample_47_T"],
		"extreme":["Sample_7_D","Sample_38_E","Sample_8_D","Sample_43_E","Sample_6_D","Sample_39_E"],
		"discriminant":["Sample_3_D","Sample_5_D","Sample_6_D","Sample_0_D","Sample_1_D","Sample_2_D"],
		"representative":["Sample_38_E","Sample_39_E","Sample_40_E","Sample_43_E","Sample_44_T","Sample_47_T"],
		"diversity":["Sample_45_T","Sample_44_T","Sample_46_T","Sample_13_D","Sample_9_D","Sample_2_D"],
		"feature":["Sample_47_T","Sample_46_T","Sample_44_T","Sample_45_T","Sample_24_R","Sample_19_R"]}
        lsKeys = ["distinct","extreme","discriminant","representative","diversity","feature"]
        sTestFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TempTestSelectFile.txt"])
        sAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"TestSelectFile.txt"])
        answer = ""

        if os.path.exists(sTestFile):
            os.remove(sTestFile)

        microPITA.funcWriteSelectionToFile(dictSelection=dictTest,xOutputFilePath=sTestFile)

        #Read in generated file and answer
        result = ""
        f = csv.reader(open(sTestFile,'r'),delimiter=ConstantsMicropita.c_outputFileDelim)
        result = [sRow for sRow in f]
        g = csv.reader(open(sAnswerFile,'r'),delimiter=ConstantsMicropita.c_outputFileDelim)
        answer = [sRow for sRow in g]

        if os.path.exists(sTestFile):
            os.remove(sTestFile)

        #Put answer in correct order
        dictresult = dict([(sRow[0], sRow[1:]) for sRow in result])
        result = [[sKey]+dictresult[sKey] for sKey in lsKeys]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

### Test funcReadSelectionFileToDictionary
    def testFuncReadSelectionFileToDictionaryForGoodCase(self):

        #Micropita object
        microPITA = MicroPITA()

        dictTest = {"distinct":["Sample_41_E","Sample_42_E","Sample_43_E","Sample_45_T","Sample_46_T","Sample_47_T"],
		"extreme":["Sample_7_D","Sample_38_E","Sample_8_D","Sample_43_E","Sample_6_D","Sample_39_E"],
		"discriminant":["Sample_3_D","Sample_5_D","Sample_6_D","Sample_0_D","Sample_1_D","Sample_2_D"],
		"representative":["Sample_38_E","Sample_39_E","Sample_40_E","Sample_43_E","Sample_44_T","Sample_47_T"],
		"diversity":["Sample_45_T","Sample_44_T","Sample_46_T","Sample_13_D","Sample_9_D","Sample_2_D"],
		"feature":["Sample_47_T","Sample_46_T","Sample_44_T","Sample_45_T","Sample_24_R","Sample_19_R"]}
        lsKeys = ["distinct","extreme","discriminant","representative","diversity","feature"]
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

        dictTest = {"distinct":["Sample_41_E","Sample_42_E","Sample_43_E","Sample_45_T","Sample_46_T","Sample_47_T"],
		"extreme":["Sample_7_D","Sample_38_E","Sample_8_D","Sample_43_E","Sample_6_D","Sample_39_E"],
		"discriminant":["Sample_3_D","Sample_5_D","Sample_6_D","Sample_0_D","Sample_1_D","Sample_2_D"],
		"representative":["Sample_38_E","Sample_39_E","Sample_40_E","Sample_43_E","Sample_44_T","Sample_47_T"],
		"diversity":["Sample_45_T","Sample_44_T","Sample_46_T","Sample_13_D","Sample_9_D","Sample_2_D"],
		"feature":["Sample_47_T","Sample_46_T","Sample_44_T","Sample_45_T","Sample_24_R","Sample_19_R"]}
        lsKeys = ["distinct","extreme","discriminant","representative","diversity","feature"]
        sTestFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"TempTestSelectFile.txt"])
        answer = "".join(["".join([sKey,str(dictTest[sKey])]) for sKey in lsKeys])

        #Get result
        microPITA.funcWriteSelectionToFile(dictSelection=dictTest,xOutputFilePath=sTestFile)
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
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
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
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
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
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
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
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        lfSelected = [False,True,False,True,True,False,False,False,False,True]

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
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
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        lfSelected = [True,False,True,False,False,True,True,True,True,False]

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
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
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        lfSelected = [True,True,True,True,True,False,False,False,False,False]

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")
        ldAverage = MicroPITA().funcGetAveragePopulation(abndTable=abndTable, lfCompress=lfSelected)

        #Get results
        ltpleDistances  = MicroPITA().funcMeasureDistanceFromLabelToAverageOtherLabel(abndTable=abndTable,
                                                                                     lfGroupOfInterest=lfSelected,
                                                                                     lfGroupOther=[not fflage for fflage in lfSelected])

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

        ltpleSelectedAnswer = [("700098986",0.5820896),("700098984",0.8626374),("700098982",1),("700098980",0.7416546),("700098988",0.6268657)]

        #Sort all results and answers
        ltpleDistances.sort(key=operator.itemgetter(0))
        ltpleSelectedAnswer.sort(key=operator.itemgetter(0))

        #Change doubles to ints so that precision issue will not throw off the tests (perserving to the 4th decimal place)
        ltpleSelected = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in ltpleDistances]
        ltpleSelectedAnswer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in ltpleSelectedAnswer]

        fError = False
        strError = ""
        for iindex, tple in enumerate(ltpleSelected):
            if not (str(ltpleSelectedAnswer[iindex]) == str(ltpleSelected[iindex])):
                fError = True
                strError +=  "".join([str(ltpleSelectedAnswer[iindex])," Did not match ",str(ltpleSelected[iindex])])

        self.assertEqual(False, fError, "".join([str(self),"::",strError,"."]))

#funcPerformDistanceSelection
    #See testFuncMeasureDistanceFromLabelToAverageOtherLabelForGoodCase for calculations in R
    def testFuncPerformDistanceSelectionForGoodCase2labels(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        sLabel = "sLabel"
        iSelect = 2

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")

        lsDisc0, lsDtnt0, lsOther0 = MicroPITA().funcPerformDistanceSelection(abndTable=abndTable,
                iSelectionCount=iSelect, sLabel=sLabel, sValueOfInterest=abndTable.funcGetMetadata(sLabel)[0])

        #Answer
        lsDisc0Answer = [("700098986",0.5820896),("700098988",0.6268657)]
        lsDtnt0Answer = [("700098982",1),("700098984",0.8626374)]
        lsOther0Answer = [("700098980",0.7416546)]

        #Sort answers and results
        lsDisc0Answer.sort(key=operator.itemgetter(0))
        lsDtnt0Answer.sort(key=operator.itemgetter(0))
        lsOther0Answer.sort(key=operator.itemgetter(0))
        lsDisc0.sort(key=operator.itemgetter(0))
        lsDtnt0.sort(key=operator.itemgetter(0))
        lsOther0.sort(key=operator.itemgetter(0))

        #Change doubles to ints so that precision issue will not throw off the tests (perserving to the 4th decimal place)
        lsDisc0Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDisc0Answer]
        lsDtnt0Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDtnt0Answer]
        lsOther0Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsOther0Answer]
        lsDisc0 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDisc0]
        lsDtnt0 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDtnt0]
        lsOther0 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsOther0]

        #Result answer pairs
        fError = False
        strError = ""
        llResultAnswers = [[lsDisc0,lsDisc0Answer], [lsDtnt0,lsDtnt0Answer], [lsOther0,lsOther0Answer]]

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

    def testFuncPerformDistanceSelectionForGoodCase3labels(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        sLabel = "sLabel3"
        iSelect = 1

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
                    cDelimiter = "\t", sMetadataID = sSampleIDSelection, sLastMetadata = sLastMetadataSelection, cFeatureNameDelimiter="|")

        lsDisc0, lsDtnt0, lsOther0 = MicroPITA().funcPerformDistanceSelection(abndTable=abndTable,
                iSelectionCount=iSelect, sLabel=sLabel, sValueOfInterest=abndTable.funcGetMetadata(sLabel)[0])

        #Answers
	#>library(vegan)
	#>Loading required package: permute
	#>This is vegan 2.0-2
	#> x1 = c(1,0,3,0,5) #'700098986'
	#> x2 = c(0,10,0,45,0) #'700098984'
	#> x3 = c(0,0,0,0,0) #'700098982'
	#> x4 = c(12,43,29,34,2)
	#> x5 = c(0,6,0,3,0)
	#> x6 = c(6,0,45,0,6)
	#> x7 = c(0,23,0,0,0)
	#> x8 = c(2,0,1,0,1)
	#> x9 = c(1,1,1,1,1)
	#> x10 = c(0,0,0,0,0)
	#> average2 = c(6.0,16.3,24.6,12.3,2.6)
	#> vegdist(rbind(average2,x1),method='bray')
	#    average2
	#x1 0.8135593
	#> vegdist(rbind(average2,x2),method='bray')
	#    average2
	#x2 0.6181507
	#> vegdist(rbind(average2,x3),method='bray')
	#   average2
	#x3        1
	#Warning message:
	#In vegdist(rbind(average2, x3), method = "bray") :
	#  you have empty rows: their dissimilarities may be meaningless in method bray
	#> average3 = c(0.75,6.0,0.5,0.25,0.5)
	#> vegdist(rbind(average3,x1),method='bray')
	#    average3
	#x1 0.7941176
	#> vegdist(rbind(average3,x2),method='bray')
	#    average3
	#x2 0.8015873
	#> vegdist(rbind(average3,x3),method='bray')
	#   average3
	#x3        1
	#Warning message:
	#In vegdist(rbind(average3, x3), method = "bray") :
	#  you have empty rows: their dissimilarities may be meaningless in method bray
	#> (0.8135593+0.7941176)/2.0
	#[1] 0.8038384
	#> (0.6181507+0.8015873)/2.0
	#[1] 0.709869
	#> (1+1)/2.0
	#[1] 1
	#Off in the 3rd place percision so rounded here
        lsDisc0Answer = [("700098984",0.7099)]
        lsDtnt0Answer = [("700098982",1)]
        lsOther0Answer = [("700098986",0.8031)]

        #Sort answers and results
        lsDisc0Answer.sort(key=operator.itemgetter(0))
        lsDtnt0Answer.sort(key=operator.itemgetter(0))
        lsOther0Answer.sort(key=operator.itemgetter(0))
        lsDisc0.sort(key=operator.itemgetter(0))
        lsDtnt0.sort(key=operator.itemgetter(0))
        lsOther0.sort(key=operator.itemgetter(0))

        #Change doubles to ints so that precision issue will not throw off the tests (perserving to the 4th decimal place)
        lsDisc0Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDisc0Answer]
        lsDtnt0Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDtnt0Answer]
        lsOther0Answer = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsOther0Answer]
        lsDisc0 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDisc0]
        lsDtnt0 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsDtnt0]
        lsOther0 = [(tplTuple[0],int(tplTuple[1]*10000)) for tplTuple in lsOther0]

        #Result answer pairs
        fError = False
        strError = ""
        llResultAnswers = [[lsDisc0,lsDisc0Answer], [lsDtnt0,lsDtnt0Answer], [lsOther0,lsOther0Answer]]

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
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        strSupervisedMetadata = "sLabel"
        fRunDistinct = True
        fRunDiscriminant = True
        iSelect = 2
        strOutputSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-input.txt"])
        strPredictSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-predict.txt"])

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
                    cDelimiter="\t", sMetadataID=sSampleIDSelection, sLastMetadata=sLastMetadataSelection, cFeatureNameDelimiter="|")

        lsOriginalSampleNames = abndTable.funcGetSampleNames()
        lsOriginalLabels = SVM.funcMakeLabels(lsOriginalSampleNames)

        dictResults = MicroPITA().funcRunSupervisedDistancesFromCentroids(abundanceTable=abndTable, fRunDistinct=fRunDistinct, fRunDiscriminant=fRunDiscriminant,
                                       xOutputSupFile=strOutputSVMFile, xPredictSupFile=strPredictSVMFile, strSupervisedMetadata=strSupervisedMetadata,
                                       iSampleSupSelectionCount=iSelect, lsOriginalSampleNames=lsOriginalSampleNames, lsOriginalLabels=lsOriginalLabels)

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
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        strSupervisedMetadata = "sLabel"
        fRunDistinct = True
        fRunDiscriminant = True
        iSelect = 2
        strOutputSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-input.txt"])
        strPredictSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-predict.txt"])
        strCorrectAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-input.txt"])

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
                    cDelimiter="\t", sMetadataID=sSampleIDSelection, sLastMetadata=sLastMetadataSelection, cFeatureNameDelimiter="|")

        lsOriginalSampleNames = abndTable.funcGetSampleNames()
        lsOriginalLabels = SVM.funcMakeLabels(lsOriginalSampleNames)

        MicroPITA().funcRunSupervisedDistancesFromCentroids(abundanceTable=abndTable, fRunDistinct=fRunDistinct, fRunDiscriminant=fRunDiscriminant,
                                       xOutputSupFile=strOutputSVMFile, xPredictSupFile=strPredictSVMFile, strSupervisedMetadata=strSupervisedMetadata,
                                       iSampleSupSelectionCount=iSelect, lsOriginalSampleNames=lsOriginalSampleNames, lsOriginalLabels=lsOriginalLabels)

        #Get answer and result
        a = csv.reader(open( strCorrectAnswerFile, 'r'),delimiter=ConstantsBreadCrumbs.c_strBreadCrumbsSVMSpace)
        strAnswer = [sRow for sRow in a]
        r = csv.reader(open( strOutputSVMFile, 'r'),delimiter=ConstantsBreadCrumbs.c_strBreadCrumbsSVMSpace)
        strResult = [sRow for sRow in r]

        self.assertEqual(str(strResult),str(strAnswer),"".join([str(self),"::Expected=",str(strAnswer),". Received=",str(strResult),"."]))

#Test for created predict file
    def testFuncRunSupervisedDistancesFromCentroidsForPredictFile(self):

        #Inputs
        strSelectionFile = "".join([ConstantsMicropitaTest.c_strTestingInput,"abridgedabundance.pcl"])
        sSampleIDSelection = "TID"
        sLastMetadataSelection = "STSite"
        strSupervisedMetadata = "sLabel"
        fRunDistinct = True
        fRunDiscriminant = True
        iSelect = 2
        strOutputSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-input.txt"])
        strPredictSVMFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-predict.txt"])
        strCorrectAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunSupervisedDistancesFromCentroidsForGoodCaseCheckingReturn-predict.txt"])

        abndTable = AbundanceTable.funcMakeFromFile(xInputFile=strSelectionFile,
                    cDelimiter="\t", sMetadataID=sSampleIDSelection, sLastMetadata=sLastMetadataSelection, cFeatureNameDelimiter="|")

        lsOriginalSampleNames = abndTable.funcGetSampleNames()
        lsOriginalLabels = SVM.funcMakeLabels(lsOriginalSampleNames)

        MicroPITA().funcRunSupervisedDistancesFromCentroids(abundanceTable=abndTable, fRunDistinct=fRunDistinct, fRunDiscriminant=fRunDiscriminant,
                                       xOutputSupFile=strOutputSVMFile, xPredictSupFile=strPredictSVMFile, strSupervisedMetadata=strSupervisedMetadata,
                                       iSampleSupSelectionCount=iSelect, lsOriginalSampleNames=lsOriginalSampleNames, lsOriginalLabels=lsOriginalLabels)

        #Get answer and result
        a = csv.reader(open( strCorrectAnswerFile, 'r'),delimiter=ConstantsBreadCrumbs.c_strBreadCrumbsSVMSpace)
        strAnswer = [sRow for sRow in a]
        r = csv.reader(open( strPredictSVMFile, 'r'),delimiter=ConstantsBreadCrumbs.c_strBreadCrumbsSVMSpace)
        strResult = [sRow for sRow in r]

        self.assertEqual(str(strResult),str(strAnswer),"".join([str(self),"::Expected=",str(strAnswer),". Received=",str(strResult),"."]))

### Test run
    ### Run all Methods
    def nntestFuncRunForGoodCase(self):
        sMethodName = "testFuncRunForGoodCase"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtreme,ConstantsMicropita.c_strRepresentative,
                        ConstantsMicropita.c_strFeature,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.TAB
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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)

        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseStratify(self):
        sMethodName = "testFuncRunForGoodCaseStratify"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtreme,ConstantsMicropita.c_strRepresentative,
                        ConstantsMicropita.c_strFeature,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.TAB
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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

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
        sIDName = "ID"
        sLastMetadataName = "Label"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtreme,ConstantsMicropita.c_strRepresentative,
                        ConstantsMicropita.c_strFeature,ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.TAB
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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseDiversity(self):
        """
        Test running just diversity
        """
        
        sMethodName = "testFuncRunForGoodCaseDiversity"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

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
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseRepresentative(self):
        """
        Test running just representative
        """
        
        sMethodName = "testFuncRunForGoodCaseRepresentative"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strRepresentative]

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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

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
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strRepresentative]

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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseExtreme(self):
        """
        Test running just extreme
        """
        
        sMethodName = "testFuncRunForGoodCaseExtreme"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strExtreme]

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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

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
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strExtreme]

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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseTargetedWithSelectionMethod(self):
        """
        Test running just targeted
        """
        sMethodName = "testFuncRunForGoodCaseTargetedWithSelectionMethod"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strFeature]

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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)

        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

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
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strFeature]

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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseTargetedWithoutSelectionMethod(self):
        """
        Test running just targeted
        """

        sMethodName = "testFuncRunForGoodCaseTargetedWithoutSelectionMethod"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLastMetadataName,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)

        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseAllUnsupervised(self):
        sMethodName = "testFuncRunForGoodCaseAllUnsupervised"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,
                        ConstantsMicropita.c_strRepresentativeDissimilarity,ConstantsMicropita.c_strFeature]

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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = microPITA.funcReadSelectionFileToDictionary(strAnswerFile)
       
        #Sort answers
        answer = str(["".join([str(key),":",str(sorted(answer[key]))]) for key in sorted(answer.keys())])
        result = str(["".join([str(key),":",str(sorted(result[key]))]) for key in sorted(result.keys())])

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def ntestFuncRunForGoodCaseAllUnsupervisedStratfied(self):
        sMethodName = "testFuncRunForGoodCaseAllUnsupervisedStratified"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtremeDissimilarity,
                        ConstantsMicropita.c_strRepresentativeDissimilarity,ConstantsMicropita.c_strFeature]

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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

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
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.TAB
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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = False
        error = False
        for sKey in result:
          if len(set(result[sKey])&set(self.dictAnswerClasses[sKey])) == len(result[sKey]):
            error = error and True
          else:
            error = True

        #Check result against answer
        self.assertEqual(str(error),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(error),"."]))

    def ntestFuncRunForGoodCaseDiscriminantMethod(self):
        sMethodName = "testFuncRunForGoodCaseDiscriminantMethod"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDiscriminant]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.TAB
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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   strFeatureSelection=sFeatureSelection,
                                   istmFeatures=strFileTaxa,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = False
        error = False
        for sKey in result:
          if len(set(result[sKey])&set(self.dictAnswerClasses[sKey])) == len(result[sKey]):
            error = error and True
          else:
            error = True

        #Check result against answer
        self.assertEqual(str(error),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(error),"."]))

    def ntestFuncRunForGoodCaseDistinctMethod(self):
        sMethodName = "testFuncRunForGoodCaseDistinctMethod"

        microPITA = MicroPITA()

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        sIDName = "ID"
        sLastMetadataName = "Label"
        cFileDelimiter = ConstantsMicropita.TAB
        cFeatureNameDelimiter = "|"
        strSelection = [ConstantsMicropita.c_strDistinct]

        #Other inputs
        strOutFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        cFileDelimiter = ConstantsMicropita.TAB
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
        result = microPITA.funcRun(strIDName=sIDName,
                                   strLastMetadataName=sLastMetadataName,
                                   istmInput=strFileAbund,
                                   ostmInputPredictFile=strInputPredictFile,
                                   ostmPredictFile=strPredictPredictFile,
                                   ostmCheckedFile=strCheckedAbndFile,
                                   ostmOutput=strOutFile,
                                   cDelimiter=cFileDelimiter,
                                   cFeatureNameDelimiter=cFeatureNameDelimiter,
                                   iCount=iUnsupervisedSelectionCount,
                                   lstrMethods=strSelection,
                                   strLabel=sLabel,
                                   strStratify=sUnsupervisedStratify)

        answer = False
        error = False
        for sKey in result:
          if len(set(result[sKey])&set(self.dictAnswerClasses[sKey])) == len(result[sKey]):
            error = error and True
          else:
            error = True

        #Check result against answer
        self.assertEqual(str(error),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(error),"."]))

####Test commandline calls to micropita script
    def testCallFromCommandlineForGoodCase(self):
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
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        sTaxaFile = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                     "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtreme,
                        ConstantsMicropita.c_strRepresentative,ConstantsMicropita.c_strFeature,
                        ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sTaxaFile+sSupervisedLabel

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strDiversity]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strRepresentative]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strExtreme]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        sTaxaFile = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                     "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strFeature]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sTaxaFile

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForGoodCaseTargetedAbundance(self):
        """
        Test commandline call for good case only targeted using the abundance-based method.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseTargetedAbundance"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForGoodCaseTargetedAbundance-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sAbundance = [ConstantsMicropita.c_strTargetedFeatureMethodArgument,ConstantsMicropita.c_strTargetedAbundance]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        sTaxaFile = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                     "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strFeature]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sTaxaFile+sAbundance

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForGoodCaseWithoutTargetedFile(self):
        """
        Test commandline call for good case without targeted file.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseWithoutTargetedFile"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testCallFromCommandlineForGoodCaseWithoutTargetedFile-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtreme,
                        ConstantsMicropita.c_strRepresentative,ConstantsMicropita.c_strFeature,
                        ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        sTaxaFile = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                     "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtreme,
                        ConstantsMicropita.c_strRepresentative,ConstantsMicropita.c_strFeature]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sTaxaFile

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]

        strSelection = [ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sSupervisedLabel

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strDiscriminant]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sSupervisedLabel

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForGoodCaseDistinct(self):
        """
        Test commandline call for good case distinct only.
        """

        sMethodName = "testCallFromCommandlineForGoodCaseDistinct"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForGoodCaseDistinctMethod-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strDistinct]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sSupervisedLabel

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

#######Test Commandline stratified
    def testCallFromCommandlineForStratifiedGoodCase(self):
        """
        Test commandline call for Stratified good case.
        """

        sMethodName = "testCallFromCommandlineForStratifiedGoodCase"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForStratifiedGoodCase-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sStratify = [ConstantsMicropita.c_strUnsupervisedStratifyMetadataArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        sTaxaFile = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                     "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtreme,
                        ConstantsMicropita.c_strRepresentative,ConstantsMicropita.c_strFeature,
                        ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sTaxaFile+sSupervisedLabel+sStratify

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForStratifiedGoodCaseDiversity(self):
        """
        Test commandline call for Stratified good case only diversity.
        """

        sMethodName = "testCallFromCommandlineForStratifiedGoodCaseDiversity"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForStratifiedGoodCaseDiversity-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strDiversity]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sStratify = [ConstantsMicropita.c_strUnsupervisedStratifyMetadataArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sStratify

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForStratifiedGoodCaseRepresentative(self):
        """
        Test commandline call for Stratified good case only representative.
        """

        sMethodName = "testCallFromCommandlineForStratifiedGoodCaseRepresentative"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForStratifiedGoodCaseRepresentative-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strRepresentative]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sStratify = [ConstantsMicropita.c_strUnsupervisedStratifyMetadataArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sStratify

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForStratifiedGoodCaseExtreme(self):
        """
        Test commandline call for Stratified good case only Extreme.
        """

        sMethodName = "testCallFromCommandlineForStratifiedGoodCaseExtreme"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForStratifiedGoodCaseExtreme-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strExtreme]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sStratify = [ConstantsMicropita.c_strUnsupervisedStratifyMetadataArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sStratify

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForStratifiedGoodCaseTargeted(self):
        """
        Test commandline call for Stratified good case only targeted.
        """

        sMethodName = "testCallFromCommandlineForStratifiedGoodCaseTargeted"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForStratifiedGoodCaseTargetedWithSelectionMethod-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        sTaxaFile = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                     "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strFeature]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sStratify = [ConstantsMicropita.c_strUnsupervisedStratifyMetadataArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sTaxaFile+sStratify

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForStratifiedGoodCaseUnsupervised(self):
        """
        Test commandline call for Stratified good case for unsupervised only.
        """

        sMethodName = "testCallFromCommandlineForStratifiedGoodCaseUnsupervised"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForStratifiedGoodCaseAllUnsupervised-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        sTaxaFile = [ConstantsMicropita.c_strTargetedSelectionFileArgument,
                     "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.taxa"])]
        strSelection = [ConstantsMicropita.c_strDiversity,ConstantsMicropita.c_strExtreme,
                        ConstantsMicropita.c_strRepresentative,ConstantsMicropita.c_strFeature]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sStratify = [ConstantsMicropita.c_strUnsupervisedStratifyMetadataArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sTaxaFile+sStratify

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForStratifiedGoodCaseSupervised(self):
        """
        Test commandline call for Stratified good case supervised methods only.
        """

        sMethodName = "testCallFromCommandlineForStratifiedGoodCaseSupervised"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForStratifiedSupervisedMethod-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]

        strSelection = [ConstantsMicropita.c_strDiscriminant,ConstantsMicropita.c_strDistinct]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]
        sStratify = [ConstantsMicropita.c_strUnsupervisedStratifyMetadataArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sSupervisedLabel+sStratify

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForStratifiedGoodCaseDiscriminant(self):
        """
        Test commandline call for Stratified good case discriminant only.
        """

        sMethodName = "testCallFromCommandlineForStratifiedGoodCaseDiscriminant"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForStratifiedGoodCaseDiscriminantMethod-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strDiscriminant]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]
        sStratify = [ConstantsMicropita.c_strUnsupervisedStratifyMetadataArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sSupervisedLabel+sStratify

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testCallFromCommandlineForStratifiedGoodCaseDistinct(self):
        """
        Test commandline call for Stratified good case distinct only.
        """

        sMethodName = "testCallFromCommandlineForStratifiedGoodCaseDistinct"

        #Commandline object
        commandLine = CommandLine()

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testFuncRunForStratifiedGoodCaseDistinctMethod-Correct.txt"])

        #Script
        strMicropitaScript = "".join([ConstantsMicropitaTest.c_strSRC,"micropita/MicroPITA.py"])

        #Required inputs
        strFileAbund = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        strOutputFile = "".join([ConstantsMicropitaTest.c_strTestingTMP,sMethodName,".txt"])
        sLastMetadata = [ConstantsMicropita.c_strLastMetadataNameArgument,"Label"]
        sSelectionCount = [ConstantsMicropita.c_strCountArgument,"3"]
        strSelection = [ConstantsMicropita.c_strDistinct]
        strSelectionMethods = []
        [strSelectionMethods.extend(["-m",strSelection]) for strSelection in strSelection]
        sSupervisedLabel = [ConstantsMicropita.c_strSupervisedLabelArgument,"Label"]
        sStratify = [ConstantsMicropita.c_strUnsupervisedStratifyMetadataArgument,"Label"]

        #Optional test
        lsOptionalArguments = []+sLastMetadata+sSelectionCount+sSupervisedLabel+sStratify

        #Build commandline list
        lsCommandline = [strMicropitaScript]+lsOptionalArguments+strSelectionMethods+[strFileAbund,strOutputFile]

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

    def testWriteToPredictFileForGoodCase(self):
        """
        Test writing to predict file.
        """
        sMethodName = "testWriteToPredictFile"

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testWriteToPredictFileForGoodCaseNoAppend_output-correct.predict"])
      
        #Inputs
        micropita = MicroPITA()

        #Abundance Table
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Label"
        cFeatureDelimiter = "|"
        abundanceTable = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        #Write file to TMP to be updated
        shutil.copyfile("".join([ConstantsMicropitaTest.c_strTestingInput+"testWriteToPredictFileForGoodCaseNoAppend_output.predict"]),
                 "".join([ConstantsMicropitaTest.c_strTestingTMP+"testWriteToPredictFileForGoodCaseNoAppend_output.predict"]))
        xPredictSupFile = "".join([ConstantsMicropitaTest.c_strTestingTMP+"testWriteToPredictFileForGoodCaseNoAppend_output.predict"])
        xInputLabelsFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"testWriteToPredictFileForGoodCaseNoAppend_input.predict"])
        dictltpleDistanceMeasurements = {"Class-Two":[("Sample_3_D","9.9"),("Sample_46_T","8.8")],"Class-One":[("Sample_2_D","7.7"),("Sample_43_E","6.6")]}
    
        micropita._writeToPredictFile(xPredictSupFile=xPredictSupFile, xInputLabelsFile=xInputLabelsFile,
                      dictltpleDistanceMeasurements=dictltpleDistanceMeasurements, abundanceTable=abundanceTable,
                      lsOriginalSampleNames=abundanceTable.funcGetSampleNames(), fFromUpdate=True)

        f = csv.reader(open(xPredictSupFile,'r'),delimiter=delimiter)
        result = [sRow for sRow in f]
        g = csv.reader(open(strAnswerFile,'r'),delimiter=delimiter)
        answer = [sRow for sRow in g]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testWriteToPredictFileForGoodCaseAllSamples(self):
        """
        Test writing to predict file.
        """
        sMethodName = "testWriteToPredictFile"

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testWriteToPredictFileForGoodCaseNoAppend_output-AllSamples-correct.predict"])
      
        #Inputs
        micropita = MicroPITA()

        #Abundance Table
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Label"
        cFeatureDelimiter = "|"
        abundanceTable = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        #Write file to TMP to be updated
        shutil.copyfile("".join([ConstantsMicropitaTest.c_strTestingInput+"testWriteToPredictFileForGoodCaseNoAppend_output.predict"]),
                 "".join([ConstantsMicropitaTest.c_strTestingTMP+"testWriteToPredictFileForGoodCaseNoAppend_output-AllSamples.predict"]))
        xPredictSupFile = "".join([ConstantsMicropitaTest.c_strTestingTMP+"testWriteToPredictFileForGoodCaseNoAppend_output-AllSamples.predict"])
        xInputLabelsFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"testWriteToPredictFileForGoodCaseNoAppend_input.predict"])
        dictltpleDistanceMeasurements = {"Class-One":[("Sample_0_D","1.0"),("Sample_1_D","1.1"),("Sample_2_D","1.2"),("Sample_4_D","1.4"),
		("Sample_5_D","1.5"),("Sample_6_D","1.6"),("Sample_9_D","1.9"),("Sample_15_D","1.15"),("Sample_20_R","1.20"),("Sample_21_R","1.21"),
		("Sample_22_R","1.22"),("Sample_23_R","1.23"),("Sample_24_R","1.24"),("Sample_26_R","1.26"),("Sample_29_R","1.29"),("Sample_34_E","1.34"),
		("Sample_35_E","1.35"),("Sample_36_E","1.36"),("Sample_37_E","1.37"),("Sample_38_E","1.38"),("Sample_40_E","1.40"),("Sample_43_E","1.43")],
		"Class-Two":[("Sample_3_D","2.3"),("Sample_7_D","2.7"),("Sample_8_D","2.8"),("Sample_10_D","2.10"),("Sample_11_D","2.11"),("Sample_12_D","2.12"),
		("Sample_13_D","2.13"),("Sample_14_D","2.14"),("Sample_16_R","2.16"),("Sample_17_R","2.17"),("Sample_18_R","2.18"),("Sample_19_R","2.19"),
		("Sample_25_R","2.25"),("Sample_27_R","2.27"),("Sample_28_R","2.28"),("Sample_30_E","2.30"),("Sample_31_E","2.31"),("Sample_32_E","2.32"),
		("Sample_33_E","2.33"),("Sample_39_E","2.39"),("Sample_41_E","2.41"),("Sample_42_E","2.42"),("Sample_44_T","2.44"),("Sample_45_T","2.45"),
		("Sample_46_T","2.46"),("Sample_47_T","2.47")]}
    
        micropita._writeToPredictFile(xPredictSupFile=xPredictSupFile, xInputLabelsFile=xInputLabelsFile,
                      dictltpleDistanceMeasurements=dictltpleDistanceMeasurements, abundanceTable=abundanceTable,
                      lsOriginalSampleNames=abundanceTable.funcGetSampleNames(), fFromUpdate=True)

        f = csv.reader(open(xPredictSupFile,'r'),delimiter=delimiter)
        result = [sRow for sRow in f]
        g = csv.reader(open(strAnswerFile,'r'),delimiter=delimiter)
        answer = [sRow for sRow in g]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testUpdatePredictFileForGoodCase(self):
        """
        Test updating predict file.
        """
        sMethodName = "testUpdatePredictFileForGoodCase"

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testUpdatePredictFileForGoodCase_output-correct.predict"])
      
        #Inputs
        micropita = MicroPITA()

        #Abundance Table
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Label"
        cFeatureDelimiter = "|"
        abundanceTable = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        #Write file to TMP to be updated
        shutil.copyfile("".join([ConstantsMicropitaTest.c_strTestingInput+"testWriteToPredictFileForGoodCaseNoAppend_output.predict"]),
                 "".join([ConstantsMicropitaTest.c_strTestingTMP+"testUpdatePredictFileForGoodCase_output.predict"]))
        xPredictSupFile = "".join([ConstantsMicropitaTest.c_strTestingTMP+"testUpdatePredictFileForGoodCase_output.predict"])
        xInputLabelsFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"testWriteToPredictFileForGoodCaseNoAppend_input.predict"])
        dictltpleDistanceMeasurements = {"Class-Two":[("Sample_3_D","9.9"),("Sample_46_T","8.8")],"Class-One":[("Sample_2_D","7.7"),("Sample_43_E","6.6")]}

        micropita._updatePredictFile(xPredictSupFile=xPredictSupFile, xInputLabelsFile=xInputLabelsFile,
                                     dictltpleDistanceMeasurements=dictltpleDistanceMeasurements,
                                     abundanceTable=abundanceTable, lsOriginalSampleNames=abundanceTable.funcGetSampleNames())

        f = csv.reader(open(xPredictSupFile,'r'),delimiter=delimiter)
        result = [sRow for sRow in f]
        g = csv.reader(open(strAnswerFile,'r'),delimiter=delimiter)
        answer = [sRow for sRow in g]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

    def testUpdatePredictFileForGoodCaseAllBut2Samples(self):
        """
        Test updating predict file. All are updated execept 30 and 35
        """
        sMethodName = "testUpdatePredictFileForGoodCaseAllBut2Samples"

        #Answer file
        strAnswerFile = "".join([ConstantsMicropitaTest.c_strTestingTruth,"testUpdatePredictFileForGoodCase_output-All2-correct.predict"])
      
        #Inputs
        micropita = MicroPITA()

        #Abundance Table
        #Inputs
        inputFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"Unbalanced48-GenNoise-0-SignalNoise-5.pcl"])
        delimiter = ConstantsMicropita.TAB
        sNameRow = "ID"
        sLastMetadata = "Label"
        cFeatureDelimiter = "|"
        abundanceTable = AbundanceTable.funcMakeFromFile(xInputFile=inputFile,
                                             cDelimiter = delimiter, sMetadataID = sNameRow,
                                             sLastMetadata = sLastMetadata, cFeatureNameDelimiter=cFeatureDelimiter)
        #Write file to TMP to be updated
        shutil.copyfile("".join([ConstantsMicropitaTest.c_strTestingInput+"testWriteToPredictFileForGoodCaseNoAppend_output.predict"]),
                 "".join([ConstantsMicropitaTest.c_strTestingTMP+"testUpdatePredictFileForGoodCase_output-All2.predict"]))
        xPredictSupFile = "".join([ConstantsMicropitaTest.c_strTestingTMP+"testUpdatePredictFileForGoodCase_output-All2.predict"])
        xInputLabelsFile = "".join([ConstantsMicropitaTest.c_strTestingInput+"testWriteToPredictFileForGoodCaseNoAppend_input.predict"])
        dictltpleDistanceMeasurements = {"Class-One":[("Sample_0_D","1.0"),("Sample_1_D","1.1"),("Sample_2_D","1.2"),("Sample_4_D","1.4"),
		("Sample_5_D","1.5"),("Sample_6_D","1.6"),("Sample_9_D","1.9"),("Sample_15_D","1.15"),("Sample_20_R","1.20"),("Sample_21_R","1.21"),
		("Sample_22_R","1.22"),("Sample_23_R","1.23"),("Sample_24_R","1.24"),("Sample_26_R","1.26"),("Sample_29_R","1.29"),("Sample_34_E","1.34"),
		("Sample_36_E","1.36"),("Sample_37_E","1.37"),("Sample_38_E","1.38"),("Sample_40_E","1.40"),("Sample_43_E","1.43")],
		"Class-Two":[("Sample_3_D","2.3"),("Sample_7_D","2.7"),("Sample_8_D","2.8"),("Sample_10_D","2.10"),("Sample_11_D","2.11"),("Sample_12_D","2.12"),
		("Sample_13_D","2.13"),("Sample_14_D","2.14"),("Sample_16_R","2.16"),("Sample_17_R","2.17"),("Sample_18_R","2.18"),("Sample_19_R","2.19"),
		("Sample_25_R","2.25"),("Sample_27_R","2.27"),("Sample_28_R","2.28"),("Sample_31_E","2.31"),("Sample_32_E","2.32"),
		("Sample_33_E","2.33"),("Sample_39_E","2.39"),("Sample_41_E","2.41"),("Sample_42_E","2.42"),("Sample_44_T","2.44"),("Sample_45_T","2.45"),
		("Sample_46_T","2.46"),("Sample_47_T","2.47")]}

        micropita._updatePredictFile(xPredictSupFile=xPredictSupFile, xInputLabelsFile=xInputLabelsFile,
                                     dictltpleDistanceMeasurements=dictltpleDistanceMeasurements,
                                     abundanceTable=abundanceTable, lsOriginalSampleNames=abundanceTable.funcGetSampleNames())

        f = csv.reader(open(xPredictSupFile,'r'),delimiter=delimiter)
        result = [sRow for sRow in f]
        g = csv.reader(open(strAnswerFile,'r'),delimiter=delimiter)
        answer = [sRow for sRow in g]

        #Check result against answer
        self.assertEqual(str(result),str(answer),"".join([str(self),"::Expected=",str(answer),". Received=",str(result),"."]))

##
#Creates a suite of tests
def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(MicroPITATest)
    return suite
