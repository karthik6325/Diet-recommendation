# api.py
from flask import Blueprint
from routes import weight_loss_endpoint, weight_gain_endpoint, healthy_endpoint


api_blueprint = Blueprint('api', __name__)

api_blueprint.route('/weight_loss', methods=['POST'])(weight_loss_endpoint)
api_blueprint.route('/weight_gain', methods=['POST'])(weight_gain_endpoint)
api_blueprint.route('/healthy', methods=['POST'])(healthy_endpoint)
