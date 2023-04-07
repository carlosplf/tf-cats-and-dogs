# TF Cats and Dogs

## API

It is possible to train and use the model for predictions through an API.

![API software schema](./schema.svg)

### Routes

`POST /model/<model_name>/train`

Using this route, you can run a model training routine. Where `<model_name>` is the name of the model to be trained, and `<number_of_epochs>` is the number of times the training should iterate over the dataset.

When this route is called, the software will create a new thread to run the training and answer the client request with the status.

#### Payload:

```
{"n_epochs":  <number_of_epochs>}
```

`POST /model/<model_name>/predict`

Use this route to ask for the `<model_name>`to predict an image if it is a Cat or Dog.


#### Payload:

```
{"upload_file":  <file.jpeg>}
```
