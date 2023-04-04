from flask import Blueprint
from flask import request
from model import api_ml_runner as ml


MAX_EPOCHS = 20

ml_router = Blueprint("ml_router", __name__)


@ml_router.route('/model/<model_name>/train', methods=["POST"])
def test_ml_runner(model_name):
    data = request.get_json()
    n_epochs = data.get("n_epochs", 0)

    if(n_epochs <= 0 or n_epochs > MAX_EPOCHS):
        return {"status": "Not running", "reason": "invalid n_epochs key"}
    else:
        if ml.run_training(model_name, n_epochs):
            return {"status": "Running"}
        else:
            return {"status": "Not running", "reason": "Can't start thread."}
