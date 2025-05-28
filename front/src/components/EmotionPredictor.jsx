import React, { useState } from 'react';
import WebcamCapture from './WebcamCapture';

const EmotionPredictor = ({ onPredict, backendUrl = 'http://localhost:5000' }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showWebcam, setShowWebcam] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
  };

  const sendImage = async (blob) => {
    const formData = new FormData();
    formData.append('file', blob, 'captura.jpg');

    try {
      const res = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Error de red: ${res.status} - ${res.statusText}`);
      }

      const data = await res.json();
      const prediction = data.prediction || data.emocion;
      onPredict(prediction);
    } catch (error) {
      console.error('Error al predecir:', error);
      setError(`Error al predecir: ${error.message}`);
      onPredict('Error al predecir.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Por favor, selecciona un archivo primero.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await sendImage(file);
    } finally {
      setLoading(false);
    }
  };

  const handleWebcamCapture = async (blob) => {
    setLoading(true);
    setError(null);
    try {
      await sendImage(blob);
    } finally {
      setLoading(false);
    }
  };

  const toggleWebcam = () => {
    setShowWebcam(!showWebcam);
    setError(null);
  };

  return (
    <div className="p-4 max-w-md mx-auto text-center">
      <h1 className="text-2xl font-bold mb-4">Analiza tu Emoción</h1>
      <div className="mb-4">
        <button
          onClick={toggleWebcam}
          className="bg-green-500 text-white p-2 rounded mr-2"
        >
          {showWebcam ? 'Cerrar Webcam' : 'Usar Webcam'}
        </button>
        <button
          onClick={() => setShowWebcam(false)}
          className="bg-blue-500 text-white p-2 rounded"
        >
          Subir Foto
        </button>
      </div>

      {showWebcam ? (
        <div className="section">
          <h2 className="text-xl mb-2">Captura en Tiempo Real</h2>
          <WebcamCapture onCapture={handleWebcamCapture} />
        </div>
      ) : (
        <div className="section">
          <h2 className="text-xl mb-2">Subir una Foto</h2>
          <form onSubmit={handleSubmit} className="mb-4">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="mb-2 border p-2 rounded w-full"
            />
            <button
              type="submit"
              className="bg-blue-500 text-white p-2 rounded disabled:bg-gray-400"
              disabled={loading}
            >
              {loading ? 'Procesando...' : 'Predecir'}
            </button>
          </form>
        </div>
      )}
      {error && <p className="text-red-500 mt-2">{error}</p>}
    </div>
  );
};

export default EmotionPredictor;