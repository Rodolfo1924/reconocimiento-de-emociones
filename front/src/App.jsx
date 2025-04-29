import React from "react";
import Home from "./pages/Home";
import WebcamCapture from "./components/WebcamCapture";
import Formulario from "./components/Formulario";

function App() {
  return (
    <div>
      <Home />
      <div style={{ marginTop: '2rem', padding: '1rem' }}>
        <h2>Formulario de carga</h2>
        <Formulario />
      </div>
      <div style={{ marginTop: '2rem', padding: '1rem' }}>
        <h2>Reconocimiento Facial de Emociones</h2>
        <WebcamCapture />
      </div>
    </div>
  );
}

export default App;
