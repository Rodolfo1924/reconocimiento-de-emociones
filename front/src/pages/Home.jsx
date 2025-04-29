// src/pages/Home.jsx
import React from "react";

const Home = () => {
  return (
    <div style={{
      backgroundColor: "#ffffff",
      minHeight: "100vh",
      fontFamily: "Arial, sans-serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "2rem"
    }}>
      <h1 style={{
        fontSize: "3rem",
        textAlign: "center",
        color: "#111",
      }}>
        Bien<span style={{ background: "linear-gradient(to right, #4e54c8, #8f94fb)", WebkitBackgroundClip: "text", color: "transparent" }}>veni</span>do
      </h1>

      <p style={{ fontSize: "1.2rem", color: "#333", marginTop: "1rem", maxWidth: "600px", textAlign: "center" }}>
        Con nuestra plataforma es muy fácil crear, compartir e interpretar emociones a partir de imágenes o videos.
      </p>

      <button style={{
        marginTop: "2rem",
        padding: "0.8rem 2rem",
        fontSize: "1rem",
        borderRadius: "6px",
        border: "none",
        backgroundColor: "#6C63FF",
        color: "white",
        cursor: "pointer",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)"
      }}>
        Empezar
      </button>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: "1rem",
        marginTop: "4rem",
        width: "100%",
        maxWidth: "1000px"
      }}>
        {["Reconocimiento Facial", "Carga de Archivos", "Estadísticas", "Emociones Detectadas", "Informes", "Soporte"].map((item, index) => (
          <div key={index} style={{
            backgroundColor: "#f8f9fa",
            padding: "1.5rem",
            borderRadius: "12px",
            textAlign: "center",
            fontWeight: "bold",
            color: "#333",
            transition: "transform 0.2s",
            cursor: "pointer"
          }}>
            {item}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Home;
