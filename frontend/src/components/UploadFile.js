import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './uploadfile.css';
import { Link, Navigate, useNavigate } from "react-router-dom";
function UploadFile() {
   
    const [filename, setFilename] = useState('')
    const [deneme, dedene] = useState('')
    const [status, setstatus] = useState('')
    const [searchText, setSearchText] = useState('');
    const [searchTextModel, setSearchTextModel] = useState('');
    const [displayArea, setDisplayArea] = useState('');
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [allFiles, setAllFiles] = useState([]);
const [arffFiles, setArffFiles] = useState([]);
const [modelFiles, setModelFiles] = useState([]);
const [files, setFiles] = useState([]);

    let api = 'http://127.0.0.1:8000/api'
    const getCookie = (name) => {
        const cookieValue = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
        return cookieValue ? cookieValue.pop() : '';
      };
    
    const saveFile = () =>{
        console.log('Button clicked')
    
        let formData = new FormData();
        formData.append("file_s", filename)
    
        let axiosConfig = {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        }
    
        console.log(formData)
        axios.post(api + '/files/', formData, axiosConfig).then(
            response =>{
                setstatus('File Uploaded Successfully')
                getFiles(); // call getFiles again after file upload
            }
        ).catch(error =>{
            console.log(error)
        })
        console.log(formData)
    }

    const getFiles = () => {
      axios.get(api + '/files/').then(
        response => {
          setFiles(response.data);
          setArffFiles(response.data.filter(file => file.file_s.endsWith('.arff')).map(file => {
            // Extract filename from file path
            const filename = file.file_s.split('/').pop();
            return { ...file, file_s: filename };
          }));
          setModelFiles(response.data.filter(file => file.file_s.endsWith('.model')).map(file => {
            // Extract filename from file path
            const filename = file.file_s.split('/').pop();
            return { ...file, file_s: filename };
          }));
        }
      ).catch(error => {
        console.log(error);
      });
    };

    const forceDownload = (response, title) =>{
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', title)
        document.body.appendChild(link)
        link.click()


    }

    const downloadWithAxios = (url, title)=>{
        axios({
            method: 'get',
            url,
            responseType: 'arraybuffer'
        }).then((response)=>{
            forceDownload(response, title)
        }).catch((error)=> console.log(error))

    }

    const handleClick = async() => {
        const csrfToken = getCookie('csrftoken');
        const arffFile = selectedFiles[0];
        const filename = arffFile.file_s.split(".")[0]; // extract filename without extension
        sessionStorage.setItem("selectedFilename", filename); // save filena
        const formData = new FormData();
        let data = selectedFiles[0].file_s
        let axiosConfig = {
            headers: {
                'X-CSRFToken': csrfToken,
                'Content-Type': 'multipart/form-data'
            }
        }
        if (selectedFiles.length === 1) {
          
          fetch(api + "/"+filename+'/arffa', {
            method: 'POST',
          })
            .then(response => response.json())
            .catch(error => console.error(error));
        
         
          
        
         
        } else if (selectedFiles.length === 2) {
          // Both ARFF and model files selected
          const formData = new FormData();
          // Both ARFF and model files selected
      formData.append("arff", selectedFiles.find(file => file.file_s.endsWith('.arff')).file_s);
      formData.append("model", selectedFiles.find(file => file.file_s.endsWith('.model')).file_s);
      console.log("Coocike is " +getCookie('csrftoken'));

      
        }
       getFiles();
      
      };
    useEffect(() => {
        getFiles();
        console.log(allFiles);
      }, []);
    
      const handleSearch = e => {
        setSearchText(e.target.value);
      };
      const handleSearchModel = e => {
        setSearchTextModel(e.target.value);
      };
    
  return (
    <div className="container-fluid" style={{backgroundColor: '#ffffff '}}>

      <div className="row">
        <div className="col-md-6">
          <h2 className="alert alert-success">File Upload Section</h2>

          <form>
            <div className="form-group">
              <label htmlFor="exampleFormControlFile1" className="float-left">
                Browse A File To Upload
              </label>
              <input type="file" onChange={e => setFilename(e.target.files[0])} className="form-control" />
            </div>

            <button type="button" onClick={saveFile} className="btn btn-primary float-left mt-2">
              Submit
            </button>
            <br />
            <br />
            <br />

            {status ? <h2>{status}</h2> : null}
          </form>
        </div>

        <div className="col-md-6">
  <h2 className="alert alert-success">List of Uploaded Files</h2>

  <div className="tables" style={{ maxHeight: "200px", overflowY: "auto" }}>
    <table className="table table-bordered mt-4">
      <thead>
        <tr>
          <th scope="col">File Title</th>
          <th scope="col">Download</th>
        </tr>
      </thead>
      <tbody>
        {files.map(file => {
          return (
            <tr key={file.id}>
              <td>{file.file_s}</td>
              <td>
                <button onClick={() => downloadWithAxios(file.file_s, file.id)} className="btn btn-success">
                  Download
                </button>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
</div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <h2 className="alert alert-success">Table .arff
          <div className="input-group mb-3, tables" style={{ maxHeight: "200px", overflowY: "auto" }} >
            <input
              type="text"
              className="form-control"
              placeholder="Search files by name"
              aria-label="Search files by name"
              aria-describedby="button-addon2"
              value={searchText}
              onChange={handleSearch}
            />
            <button className="btn btn-outline-secondary" type="button" id="button-addon2">
              Search
            </button>
          </div></h2>
          
          <table className="table table-bordered mt-4" >
            <thead>
              <tr>
                <th scope="col">File Title</th>
                <th scope="col">Select</th>
              </tr>
            </thead >
            <tbody >
              {arffFiles
  .filter(file => file.file_s && file.file_s.endsWith('.arff'))
  .filter(file => file.file_s.includes(searchText))
  .map(file => {
    return (
      <tr>
        <td>{file.file_s}</td>
        <td>
        <button
  onClick={() =>
    setSelectedFiles(prevFiles =>
      prevFiles.filter(prevFile => !prevFile.file_s.endsWith('.arff')).concat([file])
    )
  }
  className="btn btn-primary"
  disabled={selectedFiles.some(selectedFile => selectedFile.file_s === file.file_s)}
>
  Select
</button>
<button
    onClick={() =>
      setSelectedFiles((prevFiles) => prevFiles.filter((prevFile) => prevFile.file_s !== file.file_s))
    }
    className="btn btn-secondary"
    disabled={!selectedFiles.some((selectedFile) => selectedFile.file_s === file.file_s)}
  >
    Unselect
  </button>
        </td>
      </tr>
    );
  })}
            </tbody>
          </table>
        </div>

        <div className="col-md-6">
  <h2 className="alert alert-success">Table .Model</h2>
  <div className="input-group mb-3 tables" style={{ maxHeight: "200px", overflowY: "auto" }}>
    <input
      type="text"
      className="form-control"
      placeholder="Search files by name"
      aria-label="Search files by name"
      aria-describedby="button-addon2"
      value={searchTextModel}
      onChange={handleSearchModel}
    />
    <button className="btn btn-outline-secondary" type="button" id="button-addon2" onClick={handleSearchModel}>
      Search
    </button>
  </div>
  <table className="table table-bordered mt-4">
    <thead>
      <tr>
        <th scope="col">File Title</th>
        <th scope="col">Download</th>
      </tr>
    </thead>
    <tbody>
      {modelFiles
        .filter(file => file.file_s && file.file_s.endsWith('.model'))
        .filter(file => file.file_s.includes(searchTextModel))
        .map(file => (
          <tr key={file.id}>
            <td>{file.file_s}</td>
            <td>
              <button
                onClick={() =>
                  setSelectedFiles(prevFiles =>
                    prevFiles.filter(prevFile => !prevFile.file_s.endsWith('.model')).concat([file])
                  )
                }
                className="btn btn-primary"
                disabled={selectedFiles.some(selectedFile => selectedFile.file_s === file.file_s)}
              >
                Select
              </button>
              <button
                onClick={() =>
                  setSelectedFiles(prevFiles => prevFiles.filter(prevFile => prevFile.file_s !== file.file_s))
                }
                className="btn btn-secondary"
                disabled={!selectedFiles.some(selectedFile => selectedFile.file_s === file.file_s)}
              >
                Unselect
              </button>
            </td>
          </tr>
        ))}
    </tbody>
  </table>
</div>

        <div className="card mt-3">
  <div className="card-header bg-success text-white">Selected Files</div>
  <div className="card-body selected-files">
    <ul>
      {selectedFiles.map(file => (
        <li key={file.id}>
          <span className="file-icon">{file.file_s.endsWith('.arff') ? 'ARFF' : 'Model'}</span>
          <span className="file-name">{file.file_s}</span>
          <span className="file-actions">
            <button onClick={() => setSelectedFiles(prevFiles => prevFiles.filter(prevFile => prevFile.id !== file.id))} className="btn btn-secondary">
              Remove
            </button>
          </span>
        </li>
      ))}
      {/* add Learn button at the end */}
      <li>
      <button className="btn btn-success" disabled={selectedFiles.length === 0 || selectedFiles.length > 2} onClick={handleClick}>
        
  Learn
</button>
<Link to="/app" className="btn btn-success" disabled={(selectedFiles.length === 0 || selectedFiles.length > 2)}>go</Link>

      </li>
    </ul>
  </div>
</div>
      </div>
    </div>
  );
}

export default UploadFile;