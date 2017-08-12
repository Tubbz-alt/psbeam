import versioneer
from setuptools import (setup, find_packages)

setup(name        = 'psbeam',
      description = 'Image Processing Pipelines for Photon Controls and Data Systems',
      version     = versioneer.get_version(),
      cmdclass    = versioneer.get_cmdclass(),
      license     = 'BSD',
      author      = 'SLAC National Accelerator Laboratory',
      packages    = find_packages(),
      include_package_data = True      
      )
