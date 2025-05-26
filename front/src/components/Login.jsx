import React, { useState } from 'react';

const Login = ({ onLogin, onRegister }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isRegistering) {
      onRegister(email, password);
    } else {
      onLogin(email, password);
    }
  };

  const toggleForm = () => {
    setIsRegistering(!isRegistering);
    setEmail('');
    setPassword('');
  };

  return (
    <div className="container text-center">
      <h1 className="text-2xl font-bold mb-6">
        {isRegistering ? 'Crear una Cuenta' : 'Iniciar Sesión'}
      </h1>
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="mb-4">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Correo electrónico"
            className="mb-2 p-3 border rounded-lg w-full max-w-md mx-auto"
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Contraseña"
            className="mb-2 p-3 border rounded-lg w-full max-w-md mx-auto"
          />
        </div>
        <button type="submit" className="bg-blue-500 text-white p-3 rounded-lg">
          {isRegistering ? 'Registrarse' : 'Iniciar Sesión'}
        </button>
      </form>
      <p className="text-white">
        {isRegistering ? '¿Ya tienes una cuenta?' : '¿No tienes cuenta?'}{' '}
        <button
          onClick={toggleForm}
          className="text-green-500 hover:text-green-400 bg-transparent border-none"
        >
          {isRegistering ? 'Iniciar Sesión' : 'Registrarse'}
        </button>
      </p>
    </div>
  );
};

export default Login;