import React, { useState } from "react";
import './styles.css';

const Login = ({ onLogin, onRegister }) => {
  const [isLoginMode, setIsLoginMode] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isLoginMode) {
      onLogin(email, password);
    } else {
      onRegister(email, password);
    }
  };

  return (
    <div className="container" style={{ minHeight: "100vh", backgroundColor: "#f7f9fc" }}>
      <h2 style={{ fontSize: "2rem", marginBottom: "1.5rem", color: "#2E3192" }}>
        {isLoginMode ? "Iniciar Sesión" : "Registrarse"}
      </h2>
      <form onSubmit={handleSubmit} className="section">
        <div className="form-group">
          <label>Correo electrónico:</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label>Contraseña:</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" className="button">
          {isLoginMode ? "Iniciar Sesión" : "Registrarse"}
        </button>
        <p style={{ marginTop: "1rem", textAlign: "center" }}>
          {isLoginMode ? "¿No tienes cuenta?" : "¿Ya tienes cuenta?"}{" "}
          <span
            onClick={() => setIsLoginMode(!isLoginMode)}
            style={{ color: "#6C63FF", cursor: "pointer", textDecoration: "underline" }}
          >
            {isLoginMode ? "Regístrate aquí" : "Inicia sesión aquí"}
          </span>
        </p>
      </form>
    </div>
  );
};

export default Login;