import React, { useState } from "react";

const Formulario = () => {
  const [nombre, setNombre] = useState("");
  const [archivo, setArchivo] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    alert(`Nombre: ${nombre}\nArchivo: ${archivo?.name}`);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>Nombre del sujeto:</label><br />
        <input
          type="text"
          value={nombre}
          onChange={(e) => setNombre(e.target.value)}
          required
        />
      </div>
      <div style={{ marginTop: '1rem' }}>
        <label>Imagen o video:</label><br />
        <input
          type="file"
          accept="image/*,video/*"
          onChange={(e) => setArchivo(e.target.files[0])}
          required
        />
      </div>
      <button type="submit" style={{ marginTop: '1rem' }}>Enviar</button>
    </form>
  );
};

export default Formulario;
