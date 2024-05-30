import React, { useState } from 'react';
import axios from 'axios';

function ImageUpload() {
  const [image, setImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [response, setResponse] = useState(null);
  const [explanationUrl, setExplanationUrl] = useState(null); // For storing the explanation image URL

  // Handle file input change
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle form submission for prediction
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!image) {
      alert('Please select an image to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', image);  // Ensure the key matches your Flask API expectation

    try {
      const res = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResponse(res.data);  // Store the entire response data
    } catch (error) {
      console.error('Error uploading image:', error);
      alert('Failed to upload image.');
    }
  };

  // Handle form submission for explanation
  const handleExplain = async () => {
    if (!image) {
      alert('Please upload an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);

    try {
      const res = await axios.post('http://127.0.0.1:5000/explain', formData, {
        responseType: 'blob',  // Expecting a blob response (image data)
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      const url = URL.createObjectURL(res.data);
      setExplanationUrl(url);
    } catch (error) {
      console.error('Error fetching explanation:', error);
      alert('Failed to fetch explanation.');
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <button type="submit">Upload for Prediction</button>
      </form>
      {imagePreviewUrl && <img src={imagePreviewUrl} alt="Image preview" style={{ maxWidth: '500px' }} />}
      {response && (
        <div>
          <h2>Prediction: {response.prediction}</h2>
          <p>Confidence: {response.confidence ? response.confidence.toFixed(2) : 'N/A'}</p>
        </div>
      )}
      
      <button onClick={handleExplain}>Get Explanation</button>  {/* Button to trigger explanation */}
      {explanationUrl && <div>
        <h2>Model Explanation</h2>
        <img src={explanationUrl} alt="Model Explanation" style={{ maxWidth: '500px' }} />
      </div>}
    </div>
  );
}

export default ImageUpload;
