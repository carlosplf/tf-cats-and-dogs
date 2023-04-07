import logging
from flask import Blueprint
from flask import request
from model import api_ml_runner as ml


MAX_EPOCHS = 20
UPLOAD_FOLDER = "upload"

ml_router = Blueprint("ml_router", __name__)


@ml_router.route('/model/<model_name>/train', methods=["POST"])
def ml_runner_train(model_name):
    data = request.get_json()
    n_epochs = data.get("n_epochs", 0)

    if(n_epochs <= 0 or n_epochs > MAX_EPOCHS):
        return {"status": "Not running", "reason": "invalid n_epochs key"}
    else:
        pid = ml.run_training(model_name, n_epochs)
        if pid:
            return {"status": "Running", "pid": pid}
        else:
            return {"status": "Not running", "pid": pid, "reason": "Can't start thread."}


@ml_router.route('/model/<model_name>/predict', methods=["POST"])
def ml_runner_predict(model_name):
    file = request.files.get("upload_file", None)
    if file:
        logging.info("File uploaded. Storing at upload folder.")
        filename = UPLOAD_FOLDER + "/" + file.filename
        file.save(filename)
        return {
            "status": "ok",
            "probability": str(ml.run_predict(model_name, filename))
        }
    else:
        return {
            "status": "error",
            "message": "Can't read file."
        }


@ml_router.route('/model/<pid>/status', methods=["GET"])
def ml_runner_trainer_get_status(pid):
    status = ml.get_pid_status(pid)
    if status:
        return status
    else:
        return {"status": "error", "message": "check pid value"} 
