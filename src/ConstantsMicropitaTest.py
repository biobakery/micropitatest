"""
Author: Timothy Tickle
Description: Project constants.
"""

__author__ = "Timothy Tickle"
__copyright__ = "Copyright 2012"
__credits__ = ["Timothy Tickle"]
__license__ = ""
__version__ = ""
__maintainer__ = "Timothy Tickle"
__email__ = "ttickle@sph.harvard.edu"
__status__ = "Development"

class ConstantsMicropitaTest():

    #Locations
    c_strSRC = "src/"
    c_strTestingRoot = "Testing/"
    c_strTestingTMP = c_strTestingRoot+"TMP/"
    c_strTestingData = c_strTestingRoot+"Data/"
    c_strTestingTruth = c_strTestingData+"CorrectTestingResults/"
    c_strTestingInput = c_strTestingData+"AbridgedDocuments/"

    #Indexes
    PARAMETER_NAME_POSITION = 0
    PARAMETER_TYPE_POSITION = 1
    VALUE_VALUE_POSITION = 0
    VALUE_VALIDITY_POSITION = 1

    #The value to indicate a failed test case returns an exception
    RETURNS_EXCEPTION = "EXCEPT_RETURN_TYPE"

    #Value Types
    #Boolean Types
    BOOLEAN_TYPE = "BOOL"
    BOOLEAN_TEST_VALUES = [[None,False],["True",False],["False",False],[0,False],[1,False],[True,True],[False,True]]
    #String Types
    STRING_TYPE = "STRING"
    STRING_TEST_VALUES = [[None,False],[243,False],[["hello"],False],["",False],[" ",False],["     ",False],["1234567890",True],["`~!@#$%^&*()_+|}{\":?><,./;'[]\=-",True],["a",True],["qwertyuiopasdfghjklzxcvbnm",True],["QWERTYUIOPASDFGHJKLZXCVBNM",True]]
    #File Name String
    FILE_NAME_TYPE = "FILE_STRING"
    INTEGER_TYPE = "INT"
    POSITIVE_INTEGER = "P_INT"
    NEGATIVE_INTEGER = "N_INT"
    POSITIVE_INTEGER_AND_ZER0 = "PO_INT"
    NEGATIVE_INTEGER_AND_ZER0 = "NO_INT"
    FLOAT_TYPE = "FLOAT"
    POSITIVE_FLOAT = "P_FLOAT"
    NEGATIVE_FLOAT = "N_FLOAT"
    POSITIVE_FLOAT_AND_ZERO = "PO_FLOAT"
    NEGATIVE_FLOAT_AND_ZERO = "NO_FLOAT"

    ##
    #Contructor. Never needs to be used.
    def __init__(self): pass
