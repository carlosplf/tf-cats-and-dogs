import './App.css';
import { useState, useEffect } from 'react';
import { Oval } from 'react-loading-icons'


function App() {
    const [file, setFile] = useState(undefined);
    const [cat_dog, setAnimal] = useState("");

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
            setAnimal("");
        }
    }

    useEffect(()=> {
        //When some state is changed, this hook is called.
        uploadImage();
    })

    const get_answer = () => {
        if(cat_dog === "" && !file){
            return(
                <p></p>
            );
        }
        else if (cat_dog === "" && file){
            return(
                <div className="Loading">
                    <Oval strokeWidth={4}/>
                    <p>Loading...</p>
                </div>
            );
        }
        else{
            return(
                <div className="Answer">
                    <p>It's a {cat_dog}!</p>
                </div>
            );
        }
    }

    const uploadImage = () => {
        if (!file) {
            console.log("No file selected...");
            return;
        }

        if (cat_dog !== ""){
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
                <p><a className="HeaderLink" href="https://github.com/carlosplf/tf-cats-and-dogs">Model</a></p>
                <p><a className="HeaderLink" href="https://github.com/carlosplf/tf-cats-and-dogs">GitHub</a></p>
                <p><a className="HeaderLink" href="https://github.com/carlosplf/tf-cats-and-dogs">About</a></p>
            </div>
            <h1 className="Title">Cats and Dogs!</h1>
            <h2 className="SubTitle">Is it a Cat or a Dog?</h2>
            {get_answer()}
            <div>
                <label htmlFor="inputFile" className="SendFile">Select Image</label>
                <input type="file" name="file" id="inputFile" accept=".jpeg, .jpg" style={{"visibility": "hidden"}} onChange={handleFileChange}/>
            </div>
        </div>
    );
}

export default App;
