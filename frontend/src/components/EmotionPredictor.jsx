import React, { useState } from 'react';

const EmotionPredictor = ({ onPredict, backendUrl = 'http://localhost:5000' }) => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(''); // Limpia el resultado al seleccionar un nuevo archivo
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Por favor, selecciona un archivo primero.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error de red: ${response.status} - ${response.statusText}`);
      }

      const data = await response.json();
      const prediction = data.prediction || data.emocion; // Compatible con ambos formatos
      setResult(prediction);
      if (onPredict) onPredict(prediction); // Notifica al componente padre
    } catch (error) {
      console.error('Error al predecir:', error);
      setError(`Error al predecir: ${error.message}`);
      if (onPredict) onPredict('Error al predecir.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 max-w-md mx-auto text-center">
      <h1 className="text-2xl font-bold mb-4">Reconocimiento de Emociones</h1>
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
      {result && <h2 className="text-xl mt-4">Emoción detectada: {result}</h2>}
      {error && <p className="text-red-500 mt-2">{error}</p>}
    </div>
  );
};

export default EmotionPredictor;