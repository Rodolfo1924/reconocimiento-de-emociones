import React, { useState } from "react";
import './styles.css';
import Login from "./components/Login";
import EmotionPredictor from "./components/EmotionPredictor";
import axios from 'axios';
import logo from './assets/logoexpresia.png'; // Asegúrate de que el logo esté en src/assets/

const BACKEND_URL = "https://localhost:500";

const App = () => {
  const [showWelcome, setShowWelcome] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [prediction, setPrediction] = useState(null);

  const handleStart = () => {
    setShowWelcome(false);
  };

  const handleLogin = async (email, password) => {
    try {
      const res = await axios.post(`${BACKEND_URL}/api/auth/login`, {
        email,
        password,
      });

      if (res.status === 200) {
        setIsAuthenticated(true);
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
      }
    } catch (error) {
      alert(error.response?.data?.message || 'Error al registrar usuario');
    }
  };

  const handlePrediction = (result) => {
    setPrediction(result);
  };

  const handleBackToLogin = () => {
    setIsAuthenticated(false);
    setPrediction(null);
  };

  if (showWelcome) {
    return (
      <div className="welcome-container text-center">
        <img src={logo} alt="Logo de Expresia" className="logo mb-6 animate-pulse" />
        <h1 className="text-5xl font-bold mb-4 text-glow">¡Bienvenido a Expresia!</h1>
        <p className="text-xl mb-6 text-neon">Explora el Lenguaje de tus Emociones</p>
        <button
          onClick={handleStart}
          className="bg-blue-500 text-white p-4 rounded-lg hover:bg-blue-600 text-lg animate-bounce"
        >
          Empezar
        </button>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} onRegister={handleRegister} />;
  }

  return (
    <div className="container flex flex-col items-center justify-between min-h-screen">
      <div>
        {prediction && (
          <div className="prediction-result">
            <h3>Emoción detectada: {prediction}</h3>
          </div>
        )}
        <EmotionPredictor onPredict={handlePrediction} backendUrl={BACKEND_URL} />
      </div>
      <div className="mt-auto flex justify-center w-full">
        <button
          onClick={handleBackToLogin}
          className="bg-gray-500 text-white p-2 rounded"
        >
          Cerrar Sesión
        </button>
      </div>
    </div>
  );
};

export default App;
