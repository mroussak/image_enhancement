#!/usr/bin/python3
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/Flask_App")

from real_estate_editing import app as application
application.secret_key = 'real_estate_editing_flask-app-secret-key'
