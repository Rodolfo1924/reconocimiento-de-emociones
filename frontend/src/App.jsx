import React, { useState } from "react";
import './styles.css';
import Home from "./pages/Home";
import WebcamCapture from "./components/WebcamCapture";
import Formulario from "./components/Formulario";
import Login from "./components/Login";
import axios from 'axios';

const BACKEND_URL = "http://localhost:5000";

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [showMainContent, setShowMainContent] = useState(false);

  const sendImage = async (blob) => {
    try {
      const formData = new FormData();
      formData.append('file', blob, 'captura.jpg');

      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Error al comunicarse con el backend');
      }

      const data = await res.json();
      setPrediction(data.prediction);
      return data;
    } catch (error) {
      console.error('Error al enviar la imagen:', error);
      throw error;
    }
  };

  const handleLogin = async (email, password) => {
    try {
      const res = await axios.post(`${BACKEND_URL}/api/auth/login`, {
        email,
        password,
      });

      if (res.status === 200) {
        setIsAuthenticated(true);
        setShowMainContent(false);
      }
    } catch (error) {
      alert(error.response?.data?.message || 'Error al iniciar sesión');
    }
  };

  const handleRegister = async (email, password) => {
    try {
      const res = await axios.post(`${BACKEND_URL}/api/auth/register`, {
        email,
        password,
      });

      if (res.status === 201) {
        setIsAuthenticated(true);
        setShowMainContent(false);
      }
    } catch (error) {
      alert(error.response?.data?.message || 'Error al registrar usuario');
    }
  };

  const handleStart = () => {
    setShowMainContent(true);
  };

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} onRegister={handleRegister} />;
  }

  if (!showMainContent) {
    return <Home onStart={handleStart} />;
  }

  return (
    <div className="container">
      {prediction && (
        <div className="prediction-result">
          <h3>Resultado de la predicción: {prediction}</h3>
        </div>
      )}
      <div className="section">
        <h2>Formulario de carga</h2>
        <Formulario onSubmit={sendImage} />
      </div>
      <div className="section">
        <h2>Reconocimiento Facial de Emociones</h2>
        <WebcamCapture onCapture={sendImage} />
      </div>
    </div>
  );
};

export default App;