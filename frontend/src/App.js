import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [metadata, setMetadata] = useState({});
  const [inputValues, setInputValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const[resultClass, setResultClass] = useState(null);
  const api = 'http://127.0.0.1:8000/api';
  const fileName = sessionStorage.getItem("selectedFilename");
  const arffFileName = fileName + ".arff";
  const modelFileName = fileName + ".model";
  useEffect(() => {
    fetch(api+'/'+arffFileName+'/metadata')
      .then(response => response.json())
      .then(data => setMetadata(data))
      .catch(error => console.error(error));
  }, []);

  const handleInputChange = event => {
    const { name, value } = event.target;
    const attributeType = metadata[name].type.toLowerCase();

    if (attributeType === 'nominal') {
      setInputValues({ ...inputValues, [name]: value });
    } else {
      setInputValues({ ...inputValues, [name]: parseFloat(value) });
    }
  };

  const handleSubmit = event => {
    event.preventDefault();
  
    const fileName = sessionStorage.getItem("selectedFilename");
    const filename = fileName + ".arff";
  
    const formData = new FormData();
    formData.append('filename', filename);
    formData.append('input_values', JSON.stringify(inputValues));
    for (const pair of formData.entries()) {
      console.log(pair[0]+ ', ' + pair[1]);
    }
    fetch(api + "/"+filename+'/predict', {
      method: 'POST',
      body: formData
    })
      .then(response => response.json())
      .then(data => {
        setPrediction(data.result)
        setResultClass(data.last_column_name) })
     
      .catch(error => console.error(error));
  };

  return (
    <div className="container">
      <div className="card">
        <h1>Enter Input Values</h1>
        <form onSubmit={handleSubmit}>
          {Object.keys(metadata).map(key => (
            <div key={key}>
              <label htmlFor={key}>{key}</label>
              {metadata[key].type === 'nominal' ? (
                <select id={key} name={key} onChange={handleInputChange} defaultValue="">
                  <option value="">-- Select an option --</option>
                  {metadata[key].options.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              ) : (
                <input type="text" id={key} name={key} onChange={handleInputChange} />
              )}
            </div>
          ))}
          <button type="submit" disabled={Object.keys(inputValues).length === 0}>Submit</button>
        </form>

        {prediction && (
          <div>
            <h2>{resultClass}: {prediction}</h2>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;