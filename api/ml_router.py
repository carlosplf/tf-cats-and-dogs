from flask import Blueprint
from model import api_ml_runner as ml


ml_router = Blueprint("ml_router", __name__)


@ml_router.route('/ml/test')
def test_ml_runner():
    ds = ml.create_train_dataset()
    return str(len(ds))
