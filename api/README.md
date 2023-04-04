# TF Cats and Dogs

## API

It is possible to train and use the model for predictions through an API.

### Routes

`POST /model/<model_name>/train/<number_of_epochs>`

Using this route, you can run a model training routine. Where `<model_name>` is the name of the model to be trained, and `<number_of_epochs>` is the number of times the training should iterate over the dataset.

When this route is called, the software will create a new thread to run the training and answer the client request with the status.

`POST /model/<model_name>/predict/`

Use this route to ask for the `<model_name>`to predict an image if it is a Cat or Dog.

POST request expected payload: `{'image': 'image.jpg'}`
