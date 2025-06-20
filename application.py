from flask import Flask, jsonify

# Elastic Beanstalk looks for a module-level variable named `application`
# that is a callable WSGI application.
application = Flask(__name__)


@application.route("/")
def index():
    """Basic health-check endpoint."""
    return jsonify(status="ok", message="Elastic Beanstalk Flask app is running")