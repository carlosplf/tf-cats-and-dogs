import './App.css';
import { useState, useEffect } from 'react';
import { Oval } from 'react-loading-icons'
import {FaCamera} from 'react-icons/fa';
import {FaGithub} from 'react-icons/fa';
import { PongSpinner } from "react-spinners-kit";


function App() {
    const [file, setFile] = useState(undefined);
    const [cat_dog, setAnimal] = useState("");
    const [confidence, setConfidence] = useState(0.0);

    //Sentences generated with ChatGPT
    const cat_messages = [
        "Looks like we have a furry feline here! It's a cat!",
        "This image is purr-fectly classified as a cat!",
        "Congratulations, you've got a paw-some cat picture here!",
        "I'm not kitten around, this is definitely a cat!",
        "We have a meow-velous feline on our hands, it's a cat!",
        "My machine learning skills are clawsome, this is a cat!",
        "This image is the cat's whiskers, it's definitely a cat!",
        "I can cat-egorically say that this is a cat!",
        "No bones about it, this is a cat-tastic picture!",
        "Fur real, this image is a cat!"
    ];

    const dog_messages = [
        "Looks like we have a barking good image here, it's a dog!",
        "According to my calculations, this is definitely a doggo!",
        "Well, well, well, we have a woof-tastic dog picture here!",
        "This image is paws-itively classified as a dog!",
        "My machine learning skills are really going to the dogs, it's a dog!",
        "This is definitely a pup-arazzi worthy picture, it's a dog!",
        "I'm not mutt-ing around, this is a dog!",
        "This image is top dog, it's definitely a dog!",
        "It's doggone clear that this is a dog picture!",
        "Fur real, this image is a paw-some dog!",
    ];

    const loading_messages = [
        "Don't worry, we'll sniff out whether it's a cat or dog in no time!",
        "We've sent the image to our secret lab for analysis. Stay tuned for the results!",
        "Our machine learning system is currently chasing its tail while analyzing the image. Results coming soon!",
        "We're currently barking up the right tree to determine if it's a cat or dog!",
        "Hold your paws, we're fetching the results for you!",
        "Our team of highly trained AI pets are working hard to classify your image as a cat or dog.",
        "Our system is furiously analyzing your image. We'll have the answer in a jiffy!",
        "Our AI is paw-sitively excited to classify your image as a cat or dog.",
        "We're not kitten around, we take our cat-dog classifications seriously!",
        "Our AI is like a dog with a bone when it comes to classifying images. Results are on their way!"
    ]

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
            setAnimal("");
        }
    }

    //When some state is changed, this hook is called.
    useEffect(()=> {
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
                    <PongSpinner size={120} color="#FFF" loading={true} />
                    <p>{loading_messages[Math.floor(Math.random()*loading_messages.length)]}</p>
                </div>
            );
        }
        else{
            return(
                <div className="Answer">
                    <p>{chose_sentence()}</p>
                    <p id="confidence_score">Confidence score: {confidence.toFixed(2)}%</p>
                </div>
            );
        }
    }

    const chose_sentence = () =>{
        if (cat_dog === "Cat"){
            return cat_messages[Math.floor(Math.random()*cat_messages.length)];
        }
        if (cat_dog === "Dog"){
            return dog_messages[Math.floor(Math.random()*cat_messages.length)];
        }

        return "Can't tell..."
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
        data.append("Message", "Uploading image request.");

        const api_address = process.env.REACT_APP_API_ADDRESS;

        console.log(api_address);

        fetch(api_address + '/model/predict', {
                method: 'POST',
                body: data
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
            setConfidence(100-probability)
        }
        else{
            console.log("It's a DOG!");
            setAnimal("Dog");
            setConfidence(probability);
        }
    }
    
    return (
        <div className="App">
            <div className="Header">
                <p><a className="HeaderLink" href="https://github.com/carlosplf/tf-cats-and-dogs"><FaGithub/></a></p>
            </div>
            <h1 className="Title">Cats and Dogs!</h1>
            <h2 className="SubTitle">Machine Learning model to classify Cats and Dogs.</h2>
            {get_answer()}
            <div>
                <label htmlFor="inputFile" className="SendFile">
                    <FaCamera id="sendIcon"/>
                    <label htmlFor="inputFile" id="sendLabel">Send Image</label>
                </label>
                <input type="file" name="file" id="inputFile" accept=".jpeg, .jpg" style={{"visibility": "hidden"}} onChange={handleFileChange}/>
            </div>
        </div>
    );
}

export default App;
