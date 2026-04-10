from importlib.metadata import version
import logging

package_name = __name__
__version__ = version(package_name)

logger = logging.getLogger(package_name)
logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filemode="a",
                    filename="datastats_logs.logs"
                    )

logger.info(f"{package_name} version {__version__}")