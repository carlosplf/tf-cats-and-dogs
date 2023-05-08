# TF Cats and Dogs - API

It's possible to train and use the model for predictions through an API. The default model used in all routes is the VGG16.

The API routes are defined in the `ml_router.py` file.

### Model instance

When the API process starts, the ML model (SequentialModel) is instantiated. That way, when a prediction request arrives, the model is already loaded and running.

:hammer: **UNDER DEVELOPMENT:** When a TRAINING request arrives, it is necessary to create a new thread to instantiate the training of the new model, and then replace the current model, saving a backup of the current parameters.


![API software schema](./schema.svg)

### Routes

:rocket: `POST /model/train`

Using this route, you can run a model training routine. Where `<model_name>` is the name of the model to be trained, and `<number_of_epochs>` is the number of times the training should iterate over the dataset.

When this route is called, the software will create a new thread to run the training and answer the client request with the status.

``` js
PAYLOAD:

{"n_epochs":  <number_of_epochs>}
```

:rocket: `POST /model/predict`

Use this route to ask for the `<model_name>`to predict an image if it is a Cat or Dog.

``` js
PAYLOAD:

{"upload_file":  <file.jpeg>}
```

``` js
RETURN:
{
    "status": "ok" | "error",
    // if status is "ok"
    "probability": <float> (0-100),
    // if status is "error"
    "message": <error_message>
}
```

:rocket: `GET /model/<pid>/status`

Using the thread PID returned by the API, ask for the status of a thread and training process.

This route is still in dvelopment. Custom callbacks are needed in order to be able to map the progress and status of model training routines.
