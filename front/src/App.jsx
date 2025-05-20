import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [resultado, setResultado] = useState("");

  const handleChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      alert("Selecciona una imagen primero");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${process.env.REACT_APP_API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResultado(data.emociones || data.emocion || "Sin resultado");
  };

  return (
    <div style={{ padding: "2rem", textAlign: "center" }}>
      <h1>Detector de Emociones</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleChange} />
        <br /><br />
        <button type="submit">Analizar Imagen</button>
      </form>
      <div style={{ marginTop: "2rem" }}>
        <strong>Resultado:</strong> {resultado}
      </div>
    </div>
  );
}

export default App;
