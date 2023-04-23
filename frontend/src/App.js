import './App.css';
import { useState, useEffect } from 'react';


function App() {
    const [file, setFile] = useState(undefined);
    const [cat_dog, setAnimal] = useState("");

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
            setAnimal("");
            console.log("Hello!");
        }
    }

    useEffect(()=> {
        uploadImage();
    })

    const uploadImage = () => {
        if (!file) {
            console.log("No file selected...");
            return;
        }

        if (cat_dog !== ""){
            console.log("Image already sent.")
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
        .then((data) => process_api_return(data))
        .catch((err) => console.error(err));
    };

    const process_api_return = (data) => {
        const probability = parseFloat(data.probability);
        if(probability < 50.0){
            console.log("It's a CAT!");
            setAnimal("Cat");
        }
        else{
            console.log("It's a DOG!");
            setAnimal("Dog");
        }
    }
    
    return (
        <div className="App">
            <div className="Header">
                <p>Model</p>
                <p><a className="HeaderLink" href="#">GitHub</a></p>
                <p>About</p>
            </div>
            <h1 className="Title">Cats and Dogs!</h1>
            <h2 className="SubTitle">Is it a Cat or a Dog?</h2>
            <h2 className="Answer">{cat_dog}</h2>
            <div>
                <label htmlFor="inputFile" className="SendFile">Select Image</label>
                <input type="file" name="file" id="inputFile" style={{"visibility": "hidden"}} onChange={handleFileChange}/>
            </div>
        </div>
    );
}

export default App;
