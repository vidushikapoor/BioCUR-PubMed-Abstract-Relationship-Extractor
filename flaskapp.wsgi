import sys
import logging
logging.basicConfig(stream=sys.stderr)

sys.path.insert(0,'/var/www/html/flaskapp')
#sys.stdout = sys.stderr
from flaskapp import app as application
