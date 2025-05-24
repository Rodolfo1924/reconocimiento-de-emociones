import React, { useState } from "react";
import './styles.css';

const Formulario = ({ onSubmit }) => {
  const [nombre, setNombre] = useState("");
  const [archivo, setArchivo] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmitLocal = async (e) => {
    e.preventDefault();
    if (!archivo) {
      setError("Por favor, selecciona un archivo.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", archivo, archivo.name);
      await onSubmit(archivo);
      setError(null);
    } catch (err) {
      setError("Error al enviar el archivo. Intenta de nuevo.");
      console.error("Error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmitLocal} className="section">
      <div className="form-group">
        <label>Nombre del sujeto:</label>
        <input
          type="text"
          value={nombre}
          onChange={(e) => setNombre(e.target.value)}
          required
          disabled={isLoading}
        />
      </div>
      <div className="form-group">
        <label>Imagen o video:</label>
        <input
          type="file"
          accept="image/*,video/*"
          onChange={(e) => setArchivo(e.target.files[0])}
          required
          disabled={isLoading}
        />
      </div>
      <button type="submit" className="button" disabled={isLoading}>
        {isLoading ? "Enviando..." : "Enviar"}
      </button>
      {error && <p style={{ color: "#FF6F61", textAlign: "center" }}>{error}</p>}
    </form>
  );
};

export default Formulario;