import csv

def write_log(history, filename):
    print("Saving training history to CSV file...")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Val_loss', 'Val_accuracy'])
        for i in range(len(history["loss"])):
            writer.writerow([i, history["loss"][i], history["accuracy"][i], history["val_loss"][i], history["val_accuracy"][i]])
            i += 1