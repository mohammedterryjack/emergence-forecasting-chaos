from os import getenv

from dotenv import load_dotenv
from jpype import startJVM

from metrics.emergence.transfer_entropy import TransferEntropy
from metrics.emergence.integrated_information import IntegratedInformation

load_dotenv('config.env')
startJVM(getenv("PATH_JAVA_VIRTUAL_MACHINE"), f"-Djava.class.path={getenv('PATH_JAR')}")