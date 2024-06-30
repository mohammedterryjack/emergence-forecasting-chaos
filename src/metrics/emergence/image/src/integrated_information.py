"""Adapted from https://watermark.silverchair.com/013115_1_online.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAABYwwggWIBgkqhkiG9w0BBwagggV5MIIFdQIBADCCBW4GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMlkboQCTYA3u2YCrQAgEQgIIFPwDJHdKEgs_NUdT63NTv4i_I0PZlAY6-VzzIJbhxR6Y4sjGxTzFLiQz6f8DNuddfdsVb-ZQ1rPgUdi6AUto7z-qsjxzwQsvN4dgreC0u0V1b4UZZ6OrEl79XJhV_X6ozl5taBkne8Hd5eQPNUZxw8qVtRUz-G7HVomdaN_6J4Ry7_ei8Ne9u9i0PMePs-XJblsTtTwuGUO07DaDx6ABR1kmJn_6xmnMrInmzg6IQEHjtEMxu72dtHDym1Eg-ShH5PJavMsqrg5S2bGmTDGcGrSCH_8rvAPWD3lRp5ZdI3iek0kLzchNl-3WLcas4THqzHRb-WbLITIhdMdWIGCqNaovz7zqyqvHJ0muaLd3VQIBITpryGrROLP-9EIpX7zABi3Xsv7PhJRAQVkykrNDujQI7V0-1ODOBlVlfRqRpdbkTSEe-N6j4XY9aEybd2M57YqVZPuN6GG2-TzpdTNKvUkG8yQAgRAu6GM-znGhmick1Y3eQMUHlqqrsU0qFdPwghggyTioufKu-R0eSBsM4rQz_ZwOfzQieJYrCjlYWxYq9e4VOm_i7ukV-ss83kXSWraOP2AGuQprv5tLsgTeyvGRhWO1uG1gycjk2eDJYG4vVPb7nJyXdzBkud8FlO9aMbvwXAHh-jrIOtKE8TXc2ijzBsI-G4FDwjGkdAFjB_nyrjd4sC2afdjfWNQuRsC94i2m3XnwhKY77edbfcmEtf_b3HAKFk2tS5jM1sf7Sppui-HnJMdcfFqSTbCmXd-J1PAHhliGu4GINMq94f8WK17S7-SOvtYDjYfXe6GZgQelAgzCSkUO5FwgiXGoy28-iwsRHIADzfTRWI8mICnJC1rwcjBiBh0i1rfmxfku8cMLHf3PiFO489prPt95sKN88nPZIyKna92Itdy_C_VE8qotH9LKFMSPAcKQn-oGvRVmLUX1V6Y7ICB2_5JliWi3fFZYQgCs-qusVfXNwMQF6_NPF5U3dQswVWvyyPkPQV6tgGckN-tUeu-iTjTLn019lq3uMVUBQvn08NxuFPRWuDIlQ2_PvSPgiGbE16VnMAmft_wjEOJUCOlAqPEiOhd04U1jWvrQGx_ysPBI-Edi7LFDKSmlrMrvKZ6XYmBcWpAWcJ41lOVBVOB7IS4rxPjFUuWLU76Nn-IS0LDV2dSWTJGBooOBQQepS5MPwH_aT4N0JAvUApb8y4jNg2RpwY-P0qyKebMHWG9N5jIwCJS5ujhYQmLqkGjSBkWWqlQiakPBFLzdawsvYfHJpgM3NEtVLCO0EdNzE4hsmVDGA9UtWpneDcZlf81PuWWzfYUMrnP2QDSsmKczl34SZipSJkqSekXWco4DmTfYVlYd235rUveTCmnJ21yL1mj6jw1olVWg0VUFBms2ddic9GcUJnh1y_lKYsC3KdplbkxbbQERWrxsquueruzD_s6GEa5GDpF3zoX-Zoi8oRjOpzX8osbmNxNZ1DgFKpztT72p9i6m4ywsL1MwILsNN7FDfI_357yOV-y6ff7GZ4BtLW6iUclCIRr6mFba_6wPsvJJfeaOdH7IbezP0n277anH2xFRzuCW5RoRgv5ik9YoQ2WT6mHb9kAEgrASNzCyBnS40ugjB8fArBN9GVoPRwP1U463lFmDtQUZDNsHq6MvHiIp1YDS2WI9rmoQqSTIMGnpl7EDXY2VxCOSsHwyzIFk4K75LdCRJJclZPdRG-vBBMP81J3mYhWZtIiqWBkwMa6OOprSsvvUYxn7L0-R8_vI1ugslsKsi3HvBgbTEmUf7NSJ5M4YG"""

from jpype import JClass, JArray, JInt
from numpy import zeros, ndarray, array
from numpy import sum as np_sum

class IntegratedInformation:
    def __init__(
        self,
        k:int,
        base:int = 2,  
        phi_width:int = 3
    ) -> None:
        self.base = base
        self.phi_width = phi_width
        self.matrix_utils = JClass('infodynamics.utils.MatrixUtils')()
        self.metric3 = JClass('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete')(self.base**self.phi_width, k)
        self.metric1 = JClass('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete')(self.base, k)

    def _calculate(self, train_evolution:ndarray, test_evolution:ndarray) -> ndarray:
        self.metric1.addObservations(
            JArray(JArray(JInt))(train_evolution.tolist())
        )
        metric1_locals = array(self.metric1.computeLocalFromPreviousObservations(
            JArray(JArray(JInt))(test_evolution.tolist())
        ))

        _,width = train_evolution.shape
        _,test_width = test_evolution.shape

        for col in range(self.phi_width-1, width):
            aux = self.matrix_utils.computeCombinedValues(
                JArray(JArray(JInt))(train_evolution[:, col-(self.phi_width-1):col+1]), 
                self.base
            )
            self.metric3.addObservations(aux)
        
        test_aux = zeros(test_evolution.shape, dtype=int)
        for col in range(self.phi_width-1, test_width):
            test_aux[:, col] = self.matrix_utils.computeCombinedValues(
                JArray(JArray(JInt))(test_evolution[:, col-(self.phi_width-1):col+1]), 
                self.base
            )
        metric3_locals = array(self.metric3.computeLocalFromPreviousObservations(
            JArray(JArray(JInt))(test_aux.tolist())
        ))

        R = (self.phi_width-1) // 2
        locals = zeros(test_evolution.shape)
        for col in range(R, test_width-R):
            locals[:, col] = metric3_locals[:, col] - np_sum(metric1_locals[:, col-R:col+R+1], axis=1)
        return locals

    def emergence_filter(self, evolution:ndarray) -> ndarray:
        return self._calculate(train_evolution=evolution, test_evolution=evolution)
    