import './App.css';
import { ChangeEvent, useState, useEffect } from 'react';


function App() {
    const [file, setFile] = useState(undefined);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
            console.log("Hello!");
        }
        console.log(file);
    }

    useEffect(()=> {
        uploadImage();
    })

    const uploadImage = () => {
        if (!file) {
            console.log("No file selected...");
            return;
        }
            
        console.log("Sending to API...");

        const data = new FormData();

        data.append("upload_file", file);
        data.append("Test", "Test text");

        fetch('http://localhost:8080/model/vgg16/predict', {
                method: 'POST',
                body: data,
            }
        )
        .then((res) => res.json())
        .then((data) => console.log(data))
        .catch((err) => console.error(err));
    };
    
    return (
        <div className="App">
            <div className="Header">
                <p>Model</p>
                <p>GitHub</p>
                <p>About</p>
            </div>
            <h1 className="Title">Cats and Dogs!</h1>
            <h2 className="SubTitle">Is it a Cat or a Dog?</h2>
            <div>
                <label htmlFor="inputFile" className="SendFile">Select Image</label>
                <input type="file" name="file" id="inputFile" style={{"visibility": "hidden"}} onChange={handleFileChange}/>
            </div>
        </div>
    );
}

export default App;
