const express = require('express');
const bcrypt = require('bcrypt');
const User = require('../models/User');

const router = express.Router();

// Ruta para login
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Buscar usuario en la base
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ message: 'Correo o contraseña incorrectos' });
    }

    // Comparar contraseña con bcrypt
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ message: 'Correo o contraseña incorrectos' });
    }

    res.status(200).json({ message: 'Inicio de sesión exitoso' });
  } catch (error) {
    console.error('Error al iniciar sesión:', error);
    res.status(500).json({ message: 'Error al iniciar sesión' });
  }
});

module.exports = router;
