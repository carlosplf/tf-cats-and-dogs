import requests
import json
import time


BASE_URL = "http://127.0.0.1:8080"
N_EPOCHS = 1
TIME_INTERVAL = 2


def call_training():
    print("Calling training URL. Epochs: " + str(N_EPOCHS))
    path = "/model/vgg16/train"
    payload = {'n_epochs': N_EPOCHS}
    response = requests.post(url = "" + BASE_URL + path, json=payload)
    print(response.text)
    return response.text


def call_predict():
    pass
    # path = "/model/vgg16/predict"
    # files = {'upload_file': open('file.jpeg','rb')}
    # response = requests.post(url=""+url+path, files=files)


def call_get_updates(pid):
    pid = str(pid)

    print("Calling get status URL. PID: " + pid)

    path = "/model/" + pid + "/status"

    for i in range(1000000):
        response = requests.get(url = "" + BASE_URL + path)
        print(response.text)
        response_data = json.loads(response.text)
        status = response_data["status"]
        if status == "Stopped":
            break
        time.sleep(TIME_INTERVAL)

    return


if __name__ == "__main__":
    response_data = call_training()
    response_data = json.loads(response_data)
    pid = response_data["pid"]
    call_get_updates(pid)
    


